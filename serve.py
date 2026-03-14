"""Basic vLLM server for RTX 3090 (24GB VRAM).

Usage:
    python serve.py
    python serve.py --model meta-llama/Llama-3.1-8B-Instruct
    python serve.py --port 8080 --max-model-len 4096

Once running, query with:
    curl http://localhost:8000/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{"model": "cyankiwi/Qwen3.5-9B-AWQ-4bit", "messages": [{"role": "user", "content": "Hello!"}]}'

Or use the OpenAI Python client:
    from openai import OpenAI
    client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
    response = client.chat.completions.create(
        model="cyankiwi/Qwen3.5-9B-AWQ-4bit",
        messages=[{"role": "user", "content": "Hello!"}],
    )
"""

import subprocess
import sys


def main():
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", "cyankiwi/Qwen3.5-9B-AWQ-4bit",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--dtype", "half",
        "--max-model-len", "8192",
        "--gpu-memory-utilization", "0.90",
        "--quantization", "awq",
    ]

    # Pass through any extra CLI args
    cmd.extend(sys.argv[1:])

    print(f"Starting vLLM server...")
    print(f"  Command: {' '.join(cmd)}")
    subprocess.run(cmd)


if __name__ == "__main__":
    main()
