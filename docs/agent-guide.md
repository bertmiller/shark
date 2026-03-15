# Agent Guide to the Action Tape System

You are optimizing a vLLM inference server's scheduling decisions. This guide tells you everything you need to produce your first optimized action tape from a single replay log — no other documentation required.

## 1. What You're Optimizing

**Goodput** = the fraction of requests that meet their SLO.

SLO: `time_to_first_token < 2000ms`. The benchmark runs AIPerf against a Mooncake trace (~238 requests arriving over 60 seconds) and reports goodput at the end.

Your goal: produce a **sparse action tape** (JSON) that the scheduler replays, such that goodput is maximized for the full 60-second trace window (~238 requests). Build up to this incrementally — start with a few requests, get those working well, then scale up.

The **key insight** is that you can observe the traces themselves (`mooncake-traces/toolagent_trace.json`). Thus, you can optimize what actions the scheduler takes with perfect knowledge of what will happen in the future. You should make use of this capability to optimize your sparse action tape to maximize goodput.

## 2. The Feedback Loop

```
[trace] + [action tape] → benchmark.py → [replay log JSONL] + [goodput score]
               ↑                                    |
               └────── agent reasons ───────────────┘
```

**One-time setup — start the server:**
```
source vllm-venv/bin/activate
python setup.py --action-tape-log log.jsonl &
```
All commands (`python setup.py`, `python benchmark.py`, `python validate.py`, `aiperf`) require the venv to be active. Activate it once per shell session with `source vllm-venv/bin/activate`.
This loads the model and starts vLLM with the tape-driven scheduler and a control server (default port 8001) in the background. It creates `tape.json` (empty) if it doesn't already exist. Wait for the "Server ready!" message before continuing.

**Iterate incrementally — start small, scale up:**
1. Run the benchmark with a small request count:
   ```
   python benchmark.py --max-requests 3
   ```
2. Analyze the situation using the tools available to you, e.g. read `log_N.jsonl` to understand what happened, refer to the actual traces, write inline Python analysis scripts, etc. You can also run `aiperf analyze-trace <trace.jsonl>` to get summary statistics (ISL/OSL distributions, prefix reuse, cache hit rates) for a trace file.
3. Overwrite `tape.json` with targeted interventions.
4. Run `python benchmark.py` again. Repeat.

**Scaling up:** Once your tape handles a few requests well, increase the count progressively (`--max-requests 10`, `--max-requests 50`, etc.) until you reach the full trace. You can also use `--schedule-window` to control the time window (e.g., `--schedule-window 10000` for the first 10 seconds). The goal is to reach the full 60-second window (~238 requests) with high goodput.

Each run, `benchmark.py` POSTs `tape.json` to the scheduler's control server (`POST /reset`), which resets the tick counter to 0 and starts a new numbered log file. The initial run writes to `log_0.jsonl`, the next reset writes to `log_1.jsonl`, and so on. This means every benchmark run produces a clean log without restarting the server, and previous logs are preserved for comparison.

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
elapsed_ms                  — wall-clock ms since the run started (matches time_ms coordinates)

action                      — the TickAction applied this tick
  .tick                     — tick number (echoed)
  .eligible[]               — request IDs eligible for admission (cumulative priority queue)
  .prioritize[]             — request IDs moved to front of eligible queue (point-in-time)
  .deprioritize[]           — request IDs moved to back of eligible queue (point-in-time)
  .reorder_eligible[]       — full reorder: listed first, unlisted keep relative order (point-in-time)
  .preempt[]                — request IDs forced from running → waiting
  .evict[]                  — request IDs whose KV cache is freed
  .prefill_chunk_tokens     — cap on tokens processed during prefill (null = no cap)
  .decode_priority[]        — reorder running requests for decode scheduling

eligible_queue[]            — current eligible priority queue (in admission order)

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
  .kind                     — which action type failed (eligible/preempt/evict/decode_priority)
  .request_id               — the ID that was targeted
  .reason                   — why it was ignored
```

## 4. Request Identity — How to Name Requests in the Tape

Each request has two identities:

- **Internal ID**: `request.request_id` (e.g., `"chatcmpl-abc123"`) — assigned by vLLM at arrival time, unpredictable, changes between runs.
- **Stable alias**: `conv:<conversation_id>:turn:<turn_index>` — derived from trace metadata, deterministic across runs.

**Use stable aliases in your tape.** They are constructed from the trace headers (`X-Shark-Conversation-ID` + `X-Shark-Turn-Index`) and are identical every time you replay the same trace with the same random seed.

**Important: conversation IDs are generated by AIPerf, not taken directly from the trace file.** With the default `--random-seed 0`, AIPerf assigns sequential IDs like `session_000000`, `session_000001`, etc. The resulting stable aliases are `conv:session_000000:turn:0`, `conv:session_000001:turn:0`, and so on. **Do not guess these — always read them from the replay log's `action_tape_id` field.** Run a quick baseline (e.g., `--max-requests 3`) and check the log to see the actual aliases before writing your tape.

In the replay log, each request object in `waiting`/`running` includes both:

```json
{
  "request_id": "chatcmpl-abc123",
  "conversation_id": "session_000042",
  "trace_row": "session_000042",
  "turn_index": "0",
  "action_tape_id": "conv:session_000042:turn:0",
  "input_length": 412,
  "output_length": "96",
  "num_computed_tokens": 0,
  "num_output_tokens": 0,
  "trace_timestamp_ms": "5200"
}
```

Use the `action_tape_id` value (e.g., `"conv:session_000042:turn:0"`) in your tape's `eligible`, `prioritize`, `deprioritize`, `reorder_eligible`, `preempt`, `evict`, and `decode_priority` lists.

## 5. The Levers

Each action entry can specify any combination of these fields. Omitted fields (or empty lists) mean "no intervention." Actions fire based on their coordinate — `time_ms` fires when wall-clock time elapses, `on_arrival` fires when the named request first appears in the waiting queue.

### Eligible queue (cumulative)

| Lever | What it does | When to use it |
|---|---|---|
| `eligible` | Makes requests eligible for admission. Eligible requests are admitted from waiting → running as capacity allows, in queue priority order. **Eligibility is cumulative** — once a request is marked eligible, it stays eligible on all future ticks until admitted. | Control *which* requests can enter the running batch. Use `time_ms` to make requests eligible at specific times, or `on_arrival` to make them eligible immediately when they arrive. This is your primary lever. |
| `prioritize` | Moves listed requests to the **front** of the eligible queue (highest priority for admission). | Bump a request to the head of the line — e.g., a request approaching its TTFT SLO deadline. |
| `deprioritize` | Moves listed requests to the **back** of the eligible queue (lowest priority for admission). | Delay a request without removing eligibility — e.g., a large prefill you want to defer until the batch is lighter. |
| `reorder_eligible` | Full reorder of the eligible queue: listed requests move to the front in the given order, unlisted requests keep their relative order after. | Reshape the entire admission schedule at a specific point in time — e.g., after a burst of arrivals, reorder based on input length or SLO urgency. |

### Point-in-time actions

| Lever | What it does | When to use it |
|---|---|---|
| `preempt` | Moves running requests back to waiting (KV cache preserved). | Free up batch slots or token budget for higher-priority work. Preempted requests resume later without re-prefilling from scratch. |
| `evict` | Like preempt, but also frees KV cache. The request must re-prefill from zero when re-scheduled. | Reclaim KV memory when cache pressure is high and the request won't be needed soon. More aggressive than preempt. |
| `prefill_chunk_tokens` | Caps the token budget for prefill on this tick. `null` = no cap, `0` = skip prefill entirely. | Throttle prefill to protect decode latency. Large prefills starve running decodes of GPU time. |
| `decode_priority` | Reorders the running list for token scheduling. Requests listed first get tokens first. | Prioritize requests that are close to finishing or that have tight SLOs. |

## 6. Key Dynamics to Understand

**Admissions are fully manual.** Requests are only admitted if the tape has marked them as eligible. Eligibility is cumulative — once a request appears in an `eligible` list, it stays eligible and will be admitted as soon as capacity allows. Requests not marked eligible pile up in `waiting` indefinitely.

**Two coordinate systems.** Actions fire via wall-clock time (`time_ms`: milliseconds since the run started) or on request arrival (`on_arrival`: fires when the named request first enters the waiting queue). Use `time_ms` when you know the schedule in advance; use `on_arrival` for reactive admission of specific requests. Both can appear in the same tape. Use `elapsed_ms` in the replay log to correlate tape coordinates with actual timing.

**Prefill vs. decode tension.** Admitting a request triggers a prefill (processing all input tokens). A large prefill (e.g., 4000 tokens) consumes the tick's token budget and can starve running requests of decode slots. Use `prefill_chunk_tokens` to spread prefill across multiple ticks, or time admissions to avoid overlap with heavy decode load.

**KV cache is finite.** Watch `kv_utilization` in the replay log. When it approaches 1.0, new admissions will fail (allocation returns None). You must preempt or evict to free space before more eligible requests can be admitted. The scheduler will auto-preempt the lowest-priority running request if allocation fails, but relying on this is less controllable.

**TTFT starts at request arrival, not admission.** A request's time-to-first-token clock starts when AIPerf sends it (governed by trace timestamps). Every tick the request sits in `waiting` counts against its TTFT SLO. To meet `TTFT < 2000ms`, you must make requests eligible promptly after they arrive.

## 7. Tape Format

```json
{
  "actions": [
    {"time_ms": 0, "eligible": ["conv:session_000000:turn:0", "conv:session_000001:turn:0"]},
    {"time_ms": 2000, "eligible": ["conv:session_000005:turn:0"], "prefill_chunk_tokens": 512},
    {"time_ms": 3000, "prioritize": ["conv:session_000005:turn:0"]},
    {"time_ms": 5000, "preempt": ["conv:session_000000:turn:0"]},
    {"time_ms": 5000, "deprioritize": ["conv:session_000001:turn:0"]},
    {"on_arrival": "conv:session_000010:turn:0", "eligible": ["conv:session_000010:turn:0"]},
    {"time_ms": 8000, "decode_priority": ["conv:session_000003:turn:0", "conv:session_000002:turn:0"]}
  ]
}
```

Each entry must have exactly one coordinate:
- **`time_ms`** (int, >= 0): fires when wall-clock milliseconds since run start reaches this value. Multiple entries can share the same `time_ms`; all fire on the first tick that crosses the threshold.
- **`on_arrival`** (string): fires once, on the tick when the named request first appears in the waiting queue. Each `on_arrival` key must be unique. A preempted request re-entering the waiting queue does *not* re-fire its `on_arrival`.

Rules:
- The top-level object must have an `"actions"` key containing a list.
- Action fields (`eligible`, `prioritize`, `deprioritize`, `reorder_eligible`, `preempt`, `evict`, `prefill_chunk_tokens`, `decode_priority`) are optional per entry.
- When multiple actions fire on the same tick (e.g., a `time_ms` and an `on_arrival` coincide), their list fields are concatenated and `prefill_chunk_tokens` takes the minimum non-null value.
- Request IDs in lists can be either internal IDs or stable aliases (`conv:...:turn:...`). Prefer stable aliases.
- An empty tape `{"actions": []}` means no interventions — no requests become eligible, and idle detection still works.

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
