"""Thin local client for the Colab server (stdlib only: nothing heavy on your PC)."""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request


class LayerizeError(RuntimeError):
    pass


class LayerizeClient:
    def __init__(self, base_url: str | None = None, token: str | None = None):
        self.base_url = (base_url or os.environ.get("LAYERIZE_URL", "")).rstrip("/")
        self.token = token or os.environ.get("LAYERIZE_TOKEN", "")

    def configure(self, base_url: str, token: str) -> None:
        self.base_url, self.token = base_url.rstrip("/"), token

    def _req(self, method: str, path: str, body: bytes | None = None, ctype: str = "application/json"):
        if not self.base_url:
            raise LayerizeError("Not connected. Set LAYERIZE_URL/LAYERIZE_TOKEN or call connect(url, token).")
        req = urllib.request.Request(self.base_url + path, data=body, method=method)
        req.add_header("Authorization", f"Bearer {self.token}")
        req.add_header("User-Agent", "layerize-client")  # Cloudflare rejects the default urllib UA
        if body is not None:
            req.add_header("Content-Type", ctype)
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                return r.read(), dict(r.headers)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")
            try:
                detail = json.loads(detail).get("error", detail)
            except ValueError:
                pass
            raise LayerizeError(f"{exc.code}: {detail}") from None
        except urllib.error.URLError as exc:
            raise LayerizeError(f"cannot reach Colab server ({exc.reason}); is the notebook still running?") from None

    def _json(self, method, path, obj=None):
        data, _ = self._req(method, path, None if obj is None else json.dumps(obj).encode())
        return json.loads(data)

    def health(self) -> bool:
        data, _ = self._req("GET", "/healthz")
        return bool(json.loads(data).get("ok"))

    def submit(self, image_path: str) -> str:
        with open(image_path, "rb") as fh:
            data, _ = self._req("POST", "/v1/jobs", fh.read(), "application/octet-stream")
        return json.loads(data)["job_id"]

    def job(self, job_id: str) -> dict:
        return self._json("GET", f"/v1/jobs/{job_id}")

    def wait(self, job_id: str, timeout: float = 900, poll: float = 3) -> dict:
        end = time.time() + timeout
        while time.time() < end:
            info = self.job(job_id)
            if info["status"] == "ready":
                return info
            if info["status"] == "error":
                raise LayerizeError(info.get("error", "analysis failed"))
            time.sleep(poll)
        raise LayerizeError("timed out waiting for analysis")

    def process(self, image_path: str, timeout: float = 900) -> dict:
        return self.wait(self.submit(image_path), timeout)

    def search(self, job_id: str, query: str, top_k: int = 8) -> list[dict]:
        return self._json("POST", f"/v1/jobs/{job_id}/search", {"query": query, "top_k": top_k})["matches"]

    def segment(self, job_id: str, points=None, box=None) -> dict:
        return self._json("POST", f"/v1/jobs/{job_id}/segment", {"points": points, "box": box})["region"]

    def sheet(self, job_id: str, ids: list[int] | None = None, cols: int = 6, limit: int = 36) -> bytes:
        q = f"cols={cols}&limit={limit}" + (f"&ids={','.join(map(str, ids))}" if ids else "")
        return self._req("GET", f"/v1/jobs/{job_id}/sheet.png?{q}")[0]

    def layer(self, job_id: str, rid: int, matte: bool = True, canvas: bool = False) -> tuple[bytes, dict]:
        data, h = self._req("GET", f"/v1/jobs/{job_id}/layers/{rid}.png?matte={int(matte)}&canvas={int(canvas)}")
        h = {k.lower(): v for k, v in h.items()}
        pos = {"x": int(h["x-layer-x"]), "y": int(h["x-layer-y"]), "w": int(h["x-layer-w"]),
               "h": int(h["x-layer-h"]), "name": h.get("x-layer-name", f"layer_{rid}")}
        return data, pos

    def background(self, job_id: str, remove: list[int]) -> bytes:
        return self._req("GET", f"/v1/jobs/{job_id}/background.png?remove={','.join(map(str, remove))}")[0]

    def export(self, job_id: str, ids: list[int], fill: bool = True, fmt: str = "psd") -> tuple[bytes, str]:
        data, h = self._req("GET", f"/v1/jobs/{job_id}/export?ids={','.join(map(str, ids))}&fill={int(fill)}&fmt={fmt}")
        return data, {k.lower(): v for k, v in h.items()}.get("x-filename", f"layers.{fmt}")


def save_layers(client: LayerizeClient, job_id: str, ids: list[int], out_dir: str,
                canvas: bool = True, matte: bool = True) -> list[dict]:
    """Download layers as PNGs. canvas=True gives full-canvas PNGs that line up at (0,0) with no repositioning."""
    os.makedirs(out_dir, exist_ok=True)
    out = []
    for rid in ids:
        data, pos = client.layer(job_id, rid, matte=matte, canvas=canvas)
        path = os.path.join(out_dir, re.sub(r"[^A-Za-z0-9_.-]+", "_", pos["name"]) + ".png")
        with open(path, "wb") as fh:
            fh.write(data)
        out.append({"id": rid, "path": os.path.abspath(path), **pos, "canvas": canvas})
    return out
