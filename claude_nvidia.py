"""Run Claude Code for free through build.nvidia.com (NVIDIA NIM) models.

EDIT THE "USER CONFIG" SECTION BELOW: paste your API key(s) and model id(s).
Multiple keys are rotated automatically (each free key = ~40 requests/min,
so 3 keys = ~120 rpm). Get free keys at https://build.nvidia.com.

What it does:
  1. Loads your key(s) from this file, NVIDIA_API_KEY, a saved file, or a prompt
  2. Lets you pick a model (from your list + NVIDIA's live catalog)
  3. Starts a local LiteLLM proxy (port 4000) that translates Claude Code's
     Anthropic-format requests to OpenAI format and rotates across your keys
  4. Launches Claude Code pointed at the proxy

Usage:
    python claude_nvidia.py                              # interactive
    python claude_nvidia.py --model meta/llama-3.3-70b-instruct
    python claude_nvidia.py -p "one-shot prompt"

Switch models mid-session with: /model <id>   (nvidia_nim/ prefix optional)
"""

# ============================== USER CONFIG =================================
# Paste one or more NVIDIA API keys here. All keys are rotated per-request to
# spread load across the 40 requests/minute free-tier limit of each key.
API_KEYS = [
    # "nvapi-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
    # "nvapi-yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy",
]

# Model ids to show in the picker (any id from https://build.nvidia.com works;
# these have good tool-calling support, which Claude Code needs). The first
# one is the default. Add your own — nvidia_nim/ prefix is optional.
MODELS = [
    "moonshotai/kimi-k2.6",
    "deepseek-ai/deepseek-v4-pro",
    "qwen/qwen3.5-397b-a17b",
    "meta/llama-3.3-70b-instruct",
    "nvidia/llama-3.3-nemotron-super-49b-v1.5",
]
# ============================ END USER CONFIG ===============================

import argparse
import atexit
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request

NVIDIA_BASE = "https://integrate.api.nvidia.com/v1"
PORT = 4000
CLAUDE_DIR = os.path.join(os.path.expanduser("~"), ".claude")
CONFIG_PATH = os.path.join(CLAUDE_DIR, "litellm_config.yaml")
KEY_PATH = os.path.join(CLAUDE_DIR, "nvidia_api_keys.txt")


def get_api_keys() -> list:
    keys = [k.strip() for k in API_KEYS if k.strip()]
    if keys:
        print(f"Using {len(keys)} API key(s) from the script's USER CONFIG.")
        return keys
    env = os.environ.get("NVIDIA_API_KEY", "").strip()
    if env:
        # allow comma-separated keys in the env var
        return [k.strip() for k in env.split(",") if k.strip()]
    if os.path.exists(KEY_PATH):
        with open(KEY_PATH, encoding="utf-8") as f:
            keys = [line.strip() for line in f if line.strip()]
        if keys:
            print(f"Using {len(keys)} saved API key(s) from {KEY_PATH}")
            return keys
    print("No API key found. Get free keys at https://build.nvidia.com")
    print("(Enter several, one per line, for rotation past the 40 rpm limit.)")
    while not keys:
        while True:
            key = input(f"API key #{len(keys) + 1} (blank to finish): ").strip()
            if not key:
                break
            if not key.startswith("nvapi-"):
                print("  Warning: NVIDIA keys normally start with 'nvapi-'.")
            keys.append(key)
    if input("Save key(s) for next time? [Y/n]: ").strip().lower() != "n":
        os.makedirs(CLAUDE_DIR, exist_ok=True)
        with open(KEY_PATH, "w", encoding="utf-8") as f:
            f.write("\n".join(keys) + "\n")
        print(f"Saved to {KEY_PATH}")
    return keys


def normalize_model(model: str) -> str:
    """Accept ids with or without litellm-style prefixes: nvidia_nim/x, openai/x."""
    for prefix in ("nvidia_nim/", "custom_openai/", "openai/"):
        if model.startswith(prefix):
            return model[len(prefix):]
    return model


def fetch_models(api_key: str) -> list:
    req = urllib.request.Request(
        f"{NVIDIA_BASE}/models",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return sorted(m["id"] for m in json.load(r)["data"])
    except OSError as e:
        print(f"Could not fetch model list ({e}); you can still type a model id.")
        return []


def choose_model(api_key: str) -> str:
    print("\nFetching available models from build.nvidia.com ...")
    catalog = fetch_models(api_key)
    picks = [normalize_model(m) for m in MODELS]
    picks = [m for m in picks if not catalog or m in catalog] or picks
    print("\nYour models (edit MODELS at the top of this script to change):")
    for i, m in enumerate(picks, 1):
        print(f"  {i}. {m}")
    if catalog:
        print(f"  a. show all {len(catalog)} available models")
    print("  ...or type any model id directly (nvidia_nim/ prefix optional)")

    while True:
        choice = input(f"\nPick a model [1-{len(picks)}, a, or id] (default 1): ").strip()
        if not choice:
            return picks[0]
        if choice.lower() == "a" and catalog:
            for i, m in enumerate(catalog, 1):
                print(f"  {i:3d}. {m}")
            continue
        if choice.isdigit():
            n = int(choice)
            pool = picks if n <= len(picks) else catalog
            if 1 <= n <= len(pool):
                return pool[n - 1]
            print("Number out of range.")
            continue
        model = normalize_model(choice)
        if catalog and model not in catalog:
            if input(f"'{model}' is not in NVIDIA's catalog. Use it anyway? [y/N]: ").strip().lower() != "y":
                continue
        return model


def port_in_use(port: int) -> bool:
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def ensure_litellm() -> str:
    litellm = shutil.which("litellm")
    if litellm:
        return litellm
    print("litellm not found - installing (pip install 'litellm[proxy]') ...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "litellm[proxy]"])
    litellm = shutil.which("litellm")
    if not litellm:
        sys.exit("litellm still not found after install; check your PATH.")
    return litellm


def write_config(api_keys: list) -> None:
    # One wildcard deployment per API key: LiteLLM's router load-balances
    # across them and retries on 429s, rotating past the per-key 40 rpm cap.
    # custom_openai forwards ANY model id literally to NVIDIA's gateway (the
    # plain openai/ provider breaks the /v1/messages endpoint Claude Code uses).
    deployments = "".join(
        f"""
  - model_name: "*"
    litellm_params:
      model: "custom_openai/*"
      api_base: "{NVIDIA_BASE}"
      api_key: "{key}"
"""
        for key in api_keys
    )
    config = f"""
model_list:{deployments}
router_settings:
  routing_strategy: simple-shuffle
  num_retries: 3
  allowed_fails: 3
  cooldown_time: 30
litellm_settings:
  drop_params: true
"""
    os.makedirs(CLAUDE_DIR, exist_ok=True)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        f.write(config)


def start_proxy(litellm: str) -> None:
    if port_in_use(PORT):
        print(f"Port {PORT} already in use - assuming the proxy is already running.")
        return
    # PYTHONUTF8: litellm's startup banner crashes on Windows cp1252 consoles
    proc = subprocess.Popen(
        [litellm, "--config", CONFIG_PATH, "--port", str(PORT), "--host", "127.0.0.1"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env={**os.environ, "PYTHONUTF8": "1"},
    )
    atexit.register(proc.terminate)
    deadline = time.time() + 90
    while time.time() < deadline:
        if proc.poll() is not None:
            sys.exit(
                "LiteLLM proxy exited early. Run it manually to see the error:\n"
                f'  litellm --config "{CONFIG_PATH}" --port {PORT}'
            )
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health/liveliness", timeout=2)
            return
        except OSError:
            time.sleep(1)
    sys.exit("Timed out waiting for LiteLLM proxy to start.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Claude Code through build.nvidia.com models")
    parser.add_argument("--model", help="NVIDIA NIM model id (skips the interactive picker; "
                                        "nvidia_nim/ prefix optional)")
    args, claude_args = parser.parse_known_args()

    claude = shutil.which("claude")
    if not claude:
        sys.exit("claude CLI not found. Install: npm install -g @anthropic-ai/claude-code")

    api_keys = get_api_keys()
    model = normalize_model(args.model) if args.model else choose_model(api_keys[0])

    litellm = ensure_litellm()
    write_config(api_keys)

    rpm = 40 * len(api_keys)
    print(f"\nStarting LiteLLM proxy on http://127.0.0.1:{PORT} -> {model}")
    print(f"Rotating {len(api_keys)} key(s) (~{rpm} requests/min on free tier) ...")
    start_proxy(litellm)
    print("Proxy ready. Launching Claude Code (switch models anytime with /model <id>) ...\n")

    env = os.environ.copy()
    env.update({
        "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{PORT}",
        "ANTHROPIC_AUTH_TOKEN": "litellm-local",
        "ANTHROPIC_MODEL": model,
        "ANTHROPIC_SMALL_FAST_MODEL": model,
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    })
    sys.exit(subprocess.call([claude, *claude_args], env=env))


if __name__ == "__main__":
    main()
