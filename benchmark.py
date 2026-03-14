"""Benchmark: start vLLM with custom scheduler, replay Mooncake traces via aiperf.

Usage:
    source vllm-venv/bin/activate
    python benchmark.py                                    # one-command toolagent benchmark
    python benchmark.py --trace synthetic                  # synthetic trace
    python benchmark.py --trace conversation               # conversation trace
    python benchmark.py --schedule-window 30000            # 30s trace window (shorter run)
    python benchmark.py --goodput time_to_first_token:2000 # custom goodput SLO
    python benchmark.py --no-custom-scheduler              # compare against default scheduler
    python benchmark.py --server-already-running           # skip server start/stop
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL = "cyankiwi/Qwen3.5-9B-AWQ-4bit"
BASE_URL = "http://localhost:8000"
MAX_MODEL_LEN = 8192
SCHEDULER_CLS = "schedulers.custom.CustomScheduler"
TRACE_DIR = Path(__file__).parent / "mooncake-traces"

TRACES = {
    "conversation": TRACE_DIR / "conversation_trace.jsonl",
    "synthetic": TRACE_DIR / "synthetic_trace.jsonl",
    "toolagent": TRACE_DIR / "toolagent_trace.jsonl",
}

DEFAULT_GOODPUT = [
    "time_to_first_token:2000",
]
DEFAULT_TRACE = "toolagent"
# 60s trace window: ~238 requests arriving over 60s, processing tail ≈ 4 min → ~5 min total
DEFAULT_SCHEDULE_END_OFFSET = 60000
ACTION_TAPE_ENV_VAR = "SHARK_ACTION_TAPE_PATH"
ACTION_TAPE_LOG_ENV_VAR = "SHARK_ACTION_TAPE_LOG_PATH"


# ---------------------------------------------------------------------------
# Server management
# ---------------------------------------------------------------------------
def wait_for_server(timeout: int = 300) -> bool:
    import urllib.request
    import urllib.error

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            req = urllib.request.Request(f"{BASE_URL}/health")
            with urllib.request.urlopen(req, timeout=5):
                return True
        except (urllib.error.URLError, OSError):
            time.sleep(2)
    return False


def start_server(
    use_custom_scheduler: bool,
    *,
    action_tape_path: str | None = None,
    action_tape_log_path: str | None = None,
) -> subprocess.Popen:
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL,
        "--host", "0.0.0.0",
        "--port", "8000",
        "--dtype", "half",
        "--max-model-len", str(MAX_MODEL_LEN),
        "--gpu-memory-utilization", "0.90",
    ]
    if use_custom_scheduler:
        cmd.extend(["--scheduler-cls", SCHEDULER_CLS])

    env = os.environ.copy()
    project_dir = os.path.dirname(os.path.abspath(__file__))
    env["PYTHONPATH"] = project_dir + os.pathsep + env.get("PYTHONPATH", "")
    if use_custom_scheduler:
        if not action_tape_path:
            raise ValueError("action_tape_path is required when using the custom scheduler")
        env[ACTION_TAPE_ENV_VAR] = action_tape_path
        if action_tape_log_path:
            env[ACTION_TAPE_LOG_ENV_VAR] = action_tape_log_path

    print(f"Starting vLLM server (custom_scheduler={use_custom_scheduler})...")
    print(f"  Command: {' '.join(cmd)}")
    if use_custom_scheduler:
        print(f"  {ACTION_TAPE_ENV_VAR}={action_tape_path}")
        if action_tape_log_path:
            print(f"  {ACTION_TAPE_LOG_ENV_VAR}={action_tape_log_path}")
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return proc


def stop_server(proc: subprocess.Popen):
    if proc.poll() is None:
        print("\nStopping server...")
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def create_temp_noop_tape() -> str:
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        prefix="shark-noop-action-tape-",
        suffix=".json",
        delete=False,
    )
    with handle:
        json.dump({"ticks": []}, handle)
    return handle.name


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Benchmark vLLM with Mooncake traces via aiperf")
    parser.add_argument("--trace", choices=list(TRACES.keys()), default=DEFAULT_TRACE,
                        help=f"Which trace to replay (default: {DEFAULT_TRACE})")
    parser.add_argument("--schedule-window", type=int, default=DEFAULT_SCHEDULE_END_OFFSET,
                        help=f"Trace time window in ms (default: {DEFAULT_SCHEDULE_END_OFFSET})")
    parser.add_argument("--max-model-len", type=int, default=MAX_MODEL_LEN,
                        help=f"Filter requests exceeding this (default: {MAX_MODEL_LEN})")
    parser.add_argument("--no-custom-scheduler", action="store_true",
                        help="Use default scheduler instead of the custom scheduler")
    parser.add_argument("--server-already-running", action="store_true",
                        help="Skip starting/stopping server and target an existing server")
    parser.add_argument("--action-tape", type=str, default=None,
                        help="Action tape JSON to load when using the custom scheduler")
    parser.add_argument("--action-tape-log", type=str, default=None,
                        help="Optional replay log path to set via SHARK_ACTION_TAPE_LOG_PATH")
    parser.add_argument(
        "--goodput",
        nargs="+",
        metavar="METRIC:VALUE",
        default=None,
        help=(
            "Override the default AIPerf goodput SLOs. "
            "Default: time_to_first_token:2000. "
            "Example: --goodput time_to_first_token:1200 inter_token_latency:25"
        ),
    )
    args = parser.parse_args()

    trace_path = TRACES[args.trace]
    use_custom = not args.no_custom_scheduler
    effective_goodput = args.goodput or DEFAULT_GOODPUT
    proc = None
    temp_tape_path = None

    action_tape_path = args.action_tape or os.environ.get(ACTION_TAPE_ENV_VAR)
    action_tape_log_path = args.action_tape_log or os.environ.get(ACTION_TAPE_LOG_ENV_VAR)

    if use_custom and not args.server_already_running and not action_tape_path:
        temp_tape_path = create_temp_noop_tape()
        action_tape_path = temp_tape_path

    try:
        # Start server
        if not args.server_already_running:
            proc = start_server(
                use_custom,
                action_tape_path=action_tape_path,
                action_tape_log_path=action_tape_log_path,
            )
            print("Waiting for server to be ready...")
            if not wait_for_server():
                print("ERROR: Server failed to start. Last output:")
                if proc.stdout:
                    print(proc.stdout.read().decode()[-3000:])
                sys.exit(1)
            print("Server ready!\n")
        else:
            print("Using already-running server\n")
            if use_custom and action_tape_path:
                print(
                    "Note: --action-tape / SHARK_ACTION_TAPE_PATH is not applied when "
                    "--server-already-running is used."
                )

        # Build aiperf command
        scheduler_name = SCHEDULER_CLS if use_custom else "default"
        print(f"Trace: {args.trace}  |  Scheduler: {scheduler_name}")
        print(f"Schedule window: {args.schedule_window}ms")
        if use_custom and action_tape_path and not args.server_already_running:
            print(f"Action tape: {action_tape_path}")
        if use_custom and action_tape_log_path and not args.server_already_running:
            print(f"Action tape log: {action_tape_log_path}")
        print(f"Goodput SLOs: {' '.join(effective_goodput)}\n")

        aiperf_cmd = [
            sys.executable, "-m", "aiperf", "profile",
            "--model", MODEL,
            "--url", BASE_URL,
            "--endpoint-type", "chat",
            "--tokenizer", MODEL,
            "--streaming",
            "--custom-dataset-type", "mooncake-trace",
            "--input-file", str(trace_path),
            "--synthesis-max-isl", str(args.max_model_len),
            "--fixed-schedule",
            "--fixed-schedule-auto-offset",
            "--fixed-schedule-end-offset", str(args.schedule_window),
            "--extra-inputs", "ignore_eos:true",
            "--goodput", *effective_goodput,
        ]

        print(f"Running: {' '.join(aiperf_cmd)}\n")
        subprocess.run(aiperf_cmd)

    finally:
        if proc is not None:
            stop_server(proc)
        if temp_tape_path is not None:
            Path(temp_tape_path).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
