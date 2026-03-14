"""Replay Mooncake traces against a running vLLM server via AIPerf.

Assumes the server is already running (see setup.py). The scheduler
auto-reloads the action tape when it detects the file has changed,
so just overwrite the tape file and re-run this script.

Usage:
    python benchmark.py                                    # default: toolagent trace, 60s window
    python benchmark.py --trace synthetic                  # synthetic trace
    python benchmark.py --schedule-window 10000            # 10s window (faster iteration)
    python benchmark.py --goodput time_to_first_token:1200 # custom goodput SLO
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from schedulers.action_tape import load_action_tape_data

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL = "cyankiwi/Qwen3.5-9B-AWQ-4bit"
BASE_URL = "http://localhost:8000"
MAX_MODEL_LEN = 8192
TRACE_DIR = Path(__file__).parent / "mooncake-traces"

TRACES = {
    "conversation": TRACE_DIR / "conversation_trace.jsonl",
    "synthetic": TRACE_DIR / "synthetic_trace.jsonl",
    "toolagent": TRACE_DIR / "toolagent_trace.jsonl",
}

DEFAULT_GOODPUT = [
    "time_to_first_token:5000",
]
DEFAULT_TRACE = "toolagent"
DEFAULT_RANDOM_SEED = 0
# 60s trace window: ~238 requests arriving over 60s, processing tail ≈ 4 min → ~5 min total
DEFAULT_SCHEDULE_END_OFFSET = 60000


# ---------------------------------------------------------------------------
# AIPerf monkey-patch: inject X-Shark-* trace metadata headers
# ---------------------------------------------------------------------------
def _patch_aiperf_headers():
    from aiperf.endpoints.base_endpoint import BaseEndpoint

    _orig = BaseEndpoint.get_endpoint_headers

    def _patched(self, request_info):
        headers = _orig(self, request_info)
        headers["X-Shark-Conversation-ID"] = str(request_info.conversation_id)
        headers["X-Shark-Turn-Index"] = str(request_info.turn_index)
        if request_info.turns:
            turn = request_info.turns[request_info.turn_index]
            if turn.timestamp is not None:
                headers["X-Shark-Trace-Timestamp-Ms"] = str(int(turn.timestamp))
            if turn.max_tokens is not None:
                headers["X-Shark-Output-Length"] = str(turn.max_tokens)
        return headers

    BaseEndpoint.get_endpoint_headers = _patched


def _validate_tape_file(tape_path: Path) -> None:
    if not tape_path.exists():
        print(f"ERROR: tape file not found: {tape_path}", file=sys.stderr)
        sys.exit(1)

    try:
        payload = json.loads(tape_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(
            f"ERROR: invalid JSON in {tape_path}: line {exc.lineno}, "
            f"column {exc.colno}: {exc.msg}",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        load_action_tape_data(payload)
    except ValueError as exc:
        print(f"ERROR: invalid tape format in {tape_path}: {exc}", file=sys.stderr)
        sys.exit(1)


def _post_tape_reset(tape_path: Path, control_port: int) -> None:
    """POST the tape JSON to the control server's /reset endpoint."""
    tape_data = tape_path.read_bytes()
    url = f"http://localhost:{control_port}/reset"

    for attempt in range(10):
        try:
            req = urllib.request.Request(
                url, data=tape_data, method="POST",
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                if resp.status == 200:
                    print(f"Tape reset OK ({tape_path})")
                    return
                print(f"WARNING: /reset returned {resp.status}")
                return
        except urllib.error.HTTPError as exc:
            body = exc.read().decode(errors="replace")
            print(f"ERROR: /reset returned {exc.code}: {body}", file=sys.stderr)
            sys.exit(1)
        except (urllib.error.URLError, OSError):
            if attempt < 9:
                time.sleep(1)
                continue
            print(
                f"ERROR: could not reach control server at {url} after retries",
                file=sys.stderr,
            )
            sys.exit(1)


def _run_aiperf_inprocess(aiperf_args: list[str]):
    _patch_aiperf_headers()
    saved_argv = sys.argv
    try:
        sys.argv = ["aiperf"] + aiperf_args
        from aiperf.cli import app
        app()
    finally:
        sys.argv = saved_argv


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Replay Mooncake traces against a running vLLM server"
    )
    parser.add_argument(
        "--trace",
        choices=list(TRACES.keys()),
        default=DEFAULT_TRACE,
        help=f"Which trace to replay (default: {DEFAULT_TRACE})",
    )
    parser.add_argument(
        "--schedule-window",
        type=int,
        default=DEFAULT_SCHEDULE_END_OFFSET,
        help=f"Trace time window in ms (default: {DEFAULT_SCHEDULE_END_OFFSET})",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=MAX_MODEL_LEN,
        help=f"Filter requests exceeding this input length (default: {MAX_MODEL_LEN})",
    )
    parser.add_argument(
        "--action-tape",
        default="tape.json",
        help="Path to action tape JSON file. POSTs the tape to the "
             "control server before starting the replay (default: tape.json).",
    )
    parser.add_argument(
        "--control-port",
        type=int,
        default=8001,
        help="Port of the scheduler control server (default: 8001)",
    )
    parser.add_argument(
        "--goodput",
        nargs="+",
        metavar="METRIC:VALUE",
        default=None,
        help=(
            "Override the default AIPerf goodput SLOs. "
            "Default: time_to_first_token:5000. "
            "Example: --goodput time_to_first_token:1200 inter_token_latency:25"
        ),
    )
    args = parser.parse_args()

    trace_path = TRACES[args.trace]
    effective_goodput = args.goodput or DEFAULT_GOODPUT

    print(f"Trace: {args.trace}  |  Window: {args.schedule_window}ms")
    print(f"AIPerf random seed: {DEFAULT_RANDOM_SEED}")
    print(f"Goodput SLOs: {' '.join(effective_goodput)}\n")

    aiperf_args = [
        "profile",
        "--model", MODEL,
        "--url", BASE_URL,
        "--endpoint-type", "chat",
        "--tokenizer", MODEL,
        "--streaming",
        "--custom-dataset-type", "mooncake-trace",
        "--input-file", str(trace_path),
        "--random-seed", str(DEFAULT_RANDOM_SEED),
        "--synthesis-max-isl", str(args.max_model_len),
        "--fixed-schedule",
        "--fixed-schedule-auto-offset",
        "--fixed-schedule-end-offset", str(args.schedule_window),
        "--extra-inputs", "ignore_eos:true",
        "--goodput", *effective_goodput,
    ]

    if args.action_tape:
        tape_path = Path(args.action_tape)
        _validate_tape_file(tape_path)
        _post_tape_reset(tape_path, args.control_port)

    print(f"Running: aiperf {' '.join(aiperf_args)}\n")
    _run_aiperf_inprocess(aiperf_args)


if __name__ == "__main__":
    main()
