# Agent Guide to the Action Tape System

You are optimizing a vLLM inference server's scheduling decisions. This guide tells you everything you need to produce your first optimized action tape from a single replay log — no other documentation required.

## 1. What You're Optimizing

**Goodput** = the fraction of requests that meet their SLO.

SLO: `time_to_first_token < 5000ms`. The benchmark runs AIPerf against a Mooncake trace (~238 requests arriving over 60 seconds) and reports goodput at the end.

Your goal: produce a **sparse action tape** (JSON) that the scheduler replays, such that goodput is maximized for a given trace and hardware config.

The **key insight** is that you can observe the traces themselves (`mooncake-traces/toolagent_trace.json`). Thus, you can optimize what actions the scheduler takes with perfect knowledge of what will happen in the future. You should make use of this capability to optimize your sparse action tape to maximize goodput.

## 2. The Feedback Loop

```
[trace] + [action tape] → benchmark.py → [replay log JSONL] + [goodput score]
               ↑                                    |
               └────── agent reasons ───────────────┘
```

**One-time setup — start the server:**
```
python setup.py --action-tape-log log.jsonl &
```
This loads the model and starts vLLM with the tape-driven scheduler and a control server (default port 8001) in the background. It creates `tape.json` (empty) if it doesn't already exist. Wait for the "Server ready!" message before continuing.

**Iterate:**
1. Run the benchmark:
   ```
   python benchmark.py
   ```
2. Analyze the situation using the tools available to you, e.g. read `log_N.jsonl` to understand what happened, refer to the actual traces, write inline Python analysis scripts, etc.
3. Overwrite `tape.json` with targeted interventions.
4. Run `python benchmark.py` again. Repeat.

Each run, `benchmark.py` POSTs `tape.json` to the scheduler's control server (`POST /reset`), which resets the tick counter to 0 and starts a new numbered log file. The initial run writes to `log_0.jsonl`, the next reset writes to `log_1.jsonl`, and so on. This means every benchmark run produces a clean log without restarting the server, and previous logs are preserved for comparison.

Use `--schedule-window 10000` to replay only the first 10 seconds of the trace for faster turnaround during early exploration.

### Notes — your optimization journal

**You MUST maintain `notes.md` throughout the optimization process.** This file is your memory across iterations. After every benchmark run, append an entry with:
- The tape change you made and why
- The goodput result
- What you observed in the log (bottlenecks, ignored actions, queue buildup)
- What you plan to try next

Without this, you will repeat failed experiments and lose track of what works. Write notes *before* modifying the tape for the next run.

## 3. The Replay Log — What You're Reading

Each line in the JSONL log is one scheduler tick. Fields:

```
tick                        — sequential tick number (0, 1, 2, ...)

action                      — the TickAction applied this tick
  .tick                     — tick number (echoed)
  .admit[]                  — request IDs pulled from waiting → running
  .preempt[]                — request IDs forced from running → waiting
  .evict[]                  — request IDs whose KV cache is freed
  .prefill_chunk_tokens     — cap on tokens processed during prefill (null = no cap)
  .decode_priority[]        — reorder running requests for decode scheduling

tick_start_monotonic        — wall-clock timestamp (monotonic) when tick began
tick_end_monotonic          — wall-clock timestamp when tick ended

pre_state                   — scheduler state BEFORE the action
  .queue_depth              — number of waiting requests
  .num_running              — number of running requests
  .kv_utilization           — KV cache usage (0.0–1.0)
  .pause_state              — UNPAUSED / PAUSED_NEW / PAUSED_ALL
  .waiting[]                — per-request metadata for each waiting request
  .running[]                — per-request metadata for each running request

post_state                  — scheduler state AFTER the action (same shape)

scheduled_req_ids[]         — requests that actually got tokens scheduled this tick
preempted_req_ids[]         — requests that were preempted this tick
finished_req_ids[]          — requests that finished this tick
total_num_scheduled_tokens  — total tokens processed this tick

ignored_actions[]           — actions that couldn't be applied
  .kind                     — which action type failed (admit/preempt/evict/decode_priority)
  .request_id               — the ID that was targeted
  .reason                   — why it was ignored
```

## 4. Request Identity — How to Name Requests in the Tape

Each request has two identities:

- **Internal ID**: `request.request_id` (e.g., `"chatcmpl-abc123"`) — assigned by vLLM at arrival time, unpredictable, changes between runs.
- **Stable alias**: `conv:<conversation_id>:turn:<turn_index>` — derived from trace metadata, deterministic across runs.

**Use stable aliases in your tape.** They are constructed from the trace headers (`X-Shark-Conversation-ID` + `X-Shark-Turn-Index`) and are identical every time you replay the same trace.

In the replay log, each request object in `waiting`/`running` includes both:

```json
{
  "request_id": "chatcmpl-abc123",
  "conversation_id": "183",
  "trace_row": "183",
  "turn_index": "0",
  "action_tape_id": "conv:183:turn:0",
  "input_length": 412,
  "output_length": 96,
  "num_computed_tokens": 0,
  "num_output_tokens": 0,
  "trace_timestamp_ms": 5200
}
```

Use the `action_tape_id` value (e.g., `"conv:183:turn:0"`) in your tape's `admit`, `preempt`, `evict`, and `decode_priority` lists.

## 5. The Five Levers

Each tick, the tape can specify any combination of these actions. Omitted fields (or empty lists) mean "no intervention."

| Lever | What it does | When to use it |
|---|---|---|
| `admit` | Pulls specific requests from waiting → running, in the listed order. **If the tape has no action for this tick, nothing is admitted.** | Control *when* and *in what order* requests enter the running batch. This is your primary lever. |
| `preempt` | Moves running requests back to waiting (KV cache preserved). | Free up batch slots or token budget for higher-priority work. Preempted requests resume later without re-prefilling from scratch. |
| `evict` | Like preempt, but also frees KV cache. The request must re-prefill from zero when re-admitted. | Reclaim KV memory when cache pressure is high and the request won't be needed soon. More aggressive than preempt. |
| `prefill_chunk_tokens` | Caps the token budget for prefill on this tick. `null` = no cap, `0` = skip prefill entirely. | Throttle prefill to protect decode latency. Large prefills starve running decodes of GPU time. |
| `decode_priority` | Reorders the running list for token scheduling. Requests listed first get tokens first. | Prioritize requests that are close to finishing or that have tight SLOs. |

## 6. Key Dynamics to Understand

**Admissions are fully manual.** On a tick with no tape action, *nothing* is admitted from the waiting queue. Requests pile up in `waiting` until you explicitly admit them. This is by design — it gives you complete control.

**Prefill vs. decode tension.** Admitting a request triggers a prefill (processing all input tokens). A large prefill (e.g., 4000 tokens) consumes the tick's token budget and can starve running requests of decode slots. Use `prefill_chunk_tokens` to spread prefill across multiple ticks, or time admissions to avoid overlap with heavy decode load.

**KV cache is finite.** Watch `kv_utilization` in the replay log. When it approaches 1.0, new admissions will fail (allocation returns None). You must preempt or evict to free space before admitting more. The scheduler will auto-preempt the lowest-priority running request if allocation fails, but relying on this is less controllable.

**TTFT starts at request arrival, not admission.** A request's time-to-first-token clock starts when AIPerf sends it (governed by trace timestamps). Every tick the request sits in `waiting` counts against its TTFT SLO. To meet `TTFT < 5000ms`, you must admit requests promptly after they arrive.

**Ticks are not fixed-interval.** Tick duration depends on how much work the scheduler does. A tick with a large prefill takes longer than a decode-only tick. Use `tick_start_monotonic` / `tick_end_monotonic` to understand real timing.

## 7. Tape Format

```json
{
  "ticks": [
    {"tick": 12, "admit": ["conv:183:turn:0", "conv:42:turn:1"]},
    {"tick": 47, "preempt": ["conv:10:turn:0"]},
    {"tick": 48, "admit": ["conv:10:turn:0"], "prefill_chunk_tokens": 512},
    {"tick": 100, "decode_priority": ["conv:99:turn:0", "conv:50:turn:0"]}
  ]
}
```

Rules:
- **Sparse**: only include ticks where you want to intervene. Missing ticks = empty action (no admits, no preempts, no evicts).
- Each tick object must have a `"tick"` field (integer). All other fields are optional.
- Tick numbers must be unique (duplicate ticks cause a load error).
- Request IDs in lists can be either internal IDs or stable aliases (`conv:...:turn:...`). Prefer stable aliases.
- The top-level object must have a `"ticks"` key containing a list, or the JSON root can be a bare list of tick objects.

## 8. Your constraints

You are the **tape author**, not the system developer. You control scheduling decisions through the tape — you do not change the system that executes them.

**You CAN:**
- Edit `tape.json` — this is your primary output
- Read and analyze replay logs (`log_*.jsonl`)
- Read the trace files in `mooncake-traces/`
- Write and run analysis scripts (Python, jq, etc.) to understand logs and traces
- Maintain `notes.md` with your observations and plans

**You CANNOT:**
- Modify the scheduler (`schedulers/`), vLLM, `benchmark.py`, `setup.py`, or any system code
- Change server configuration or model parameters
- Interact with the vLLM server directly (only through `benchmark.py`)

## 9. Other considerations
### Non-interactive execution
- This session is non-interactive. No user replies are available during a run.
- Do not ask for direction, options, approval, or clarification.
- Make reasonable assumptions, choose a concrete path, implement it, and report results.

### TIMEOUTS
- The smoke test should finish quickly. If it hangs, kill it and debug before doing larger runs.
- Quick-gate runs should be short. If a benchmark is taking far longer than expected for the chosen request count, kill it and treat that as a failure signal.
- Do not leave long accidental runs chewing through the full trace unless you intentionally promoted the experiment.
