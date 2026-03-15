"""Start the vLLM server with the tape-driven custom scheduler.

Run this once in a dedicated terminal, then use benchmark.py to replay traces.

Usage:
    source vllm-venv/bin/activate
    python setup.py --action-tape tape.json --action-tape-log log.jsonl
"""

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL = "cyankiwi/Qwen3.5-9B-AWQ-4bit"
BASE_URL = "http://localhost:8000"
MAX_MODEL_LEN = 8192
SCHEDULER_CLS = "schedulers.custom.CustomScheduler"
ACTION_TAPE_ENV_VAR = "SHARK_ACTION_TAPE_PATH"
ACTION_TAPE_LOG_ENV_VAR = "SHARK_ACTION_TAPE_LOG_PATH"
CONTROL_PORT_ENV_VAR = "SHARK_CONTROL_PORT"


def wait_for_server(timeout: int = 300) -> bool:
    import urllib.error
    import urllib.request

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            req = urllib.request.Request(f"{BASE_URL}/health")
            with urllib.request.urlopen(req, timeout=5):
                return True
        except (urllib.error.URLError, OSError):
            time.sleep(2)
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Start vLLM server with tape-driven scheduler"
    )
    parser.add_argument(
        "--action-tape",
        default="tape.json",
        help="Path to action tape JSON file (default: tape.json, created if missing)",
    )
    parser.add_argument(
        "--action-tape-log",
        default=None,
        help="Path for the replay log JSONL file",
    )
    parser.add_argument(
        "--control-port",
        type=int,
        default=8001,
        help="Port for the scheduler control server (default: 8001)",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=MAX_MODEL_LEN,
        help=f"Max model context length (default: {MAX_MODEL_LEN})",
    )
    args = parser.parse_args()

    # Ensure the tape file exists so the scheduler can start.
    tape_path = Path(args.action_tape)
    if not tape_path.exists():
        tape_path.write_text('{"actions": []}', encoding="utf-8")
        print(f"Created empty tape: {tape_path}")

    action_tape_path = str(tape_path.resolve())
    action_tape_log_path = (
        str(Path(args.action_tape_log).resolve()) if args.action_tape_log else None
    )

    project_dir = os.path.dirname(os.path.abspath(__file__))
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model", MODEL,
        "--host", "0.0.0.0",
        "--port", "8000",
        "--dtype", "half",
        "--max-model-len", str(args.max_model_len),
        "--gpu-memory-utilization", "0.90",
        "--scheduler-cls", SCHEDULER_CLS,
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = project_dir + os.pathsep + env.get("PYTHONPATH", "")
    env[ACTION_TAPE_ENV_VAR] = action_tape_path
    if action_tape_log_path:
        env[ACTION_TAPE_LOG_ENV_VAR] = action_tape_log_path
    env[CONTROL_PORT_ENV_VAR] = str(args.control_port)

    print("Starting vLLM server with tape-driven scheduler...")
    print(f"  Model: {MODEL}")
    print(f"  Action tape: {action_tape_path}")
    if action_tape_log_path:
        print(f"  Replay log: {action_tape_log_path}")
    print(f"  Control port: {args.control_port}")
    print(f"  Command: {' '.join(cmd)}\n")

    proc = subprocess.Popen(cmd, cwd=project_dir, env=env)

    print("Waiting for server to be ready...")
    if not wait_for_server():
        print("ERROR: Server failed to start.")
        proc.kill()
        proc.wait()
        sys.exit(1)

    print("Server ready! Run benchmark.py in another terminal. Ctrl+C to stop.\n")

    try:
        proc.wait()
    except KeyboardInterrupt:
        print("\nStopping server...")
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


if __name__ == "__main__":
    main()
