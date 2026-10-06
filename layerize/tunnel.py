"""Expose the Colab server through a free Cloudflare quick tunnel (no account needed)."""
from __future__ import annotations

import os
import re
import stat
import subprocess
import time
import urllib.request

BIN = "/tmp/cloudflared"
URL = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"


def start(port: int, timeout: float = 60.0) -> tuple[str, subprocess.Popen]:
    if not os.path.exists(BIN):
        urllib.request.urlretrieve(URL, BIN)
        os.chmod(BIN, os.stat(BIN).st_mode | stat.S_IEXEC)
    proc = subprocess.Popen([BIN, "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{port}"],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    end = time.time() + timeout
    while time.time() < end:
        line = proc.stdout.readline()
        m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", line or "")
        if m:
            return m.group(0), proc
        if proc.poll() is not None:
            break
    proc.kill()
    raise RuntimeError("cloudflared did not give a public URL")
