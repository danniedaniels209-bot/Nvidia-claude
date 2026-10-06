"""HTTP API that runs on the Colab GPU (stdlib only, no web framework needed).

POST /v1/jobs                       body = raw image bytes  -> {"job_id"}   (analysis runs in background)
GET  /v1/jobs/{id}                  -> status + manifest of regions
GET  /v1/jobs/{id}/sheet.png        ?ids=1,2,3&cols=6       numbered contact sheet
POST /v1/jobs/{id}/search           {"query": "...", "top_k": 8}
POST /v1/jobs/{id}/segment          {"points": [[x,y,1]], "box": [x0,y0,x1,y1]}  -> new region
GET  /v1/jobs/{id}/layers/{rid}.png ?matte=1&canvas=0       RGBA cutout (X-Layer-* headers give position)
GET  /v1/jobs/{id}/background.png   ?remove=1,2             image with those regions removed + filled
GET  /v1/jobs/{id}/export           ?ids=1,2&fill=1&fmt=psd|zip
Every request except /healthz needs `Authorization: Bearer <token>`.
"""
from __future__ import annotations

import argparse
import io
import json
import re
import secrets
import threading
import traceback
import uuid
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image

from .pipeline import Job

MAX_UPLOAD = 100 * 1024 * 1024
MAX_SIDE = 4096
MAX_JOBS = 8


class State:
    def __init__(self, engines, token: str):
        self.engines, self.token = engines, token
        self.jobs: OrderedDict[str, Job] = OrderedDict()
        self.gpu = threading.Lock()  # one GPU job at a time

    def new_job(self, data: bytes) -> Job:
        im = Image.open(io.BytesIO(data))
        im = im.convert("RGB")
        if max(im.size) > MAX_SIDE:
            im.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
        job = Job(np.asarray(im), self.engines, uuid.uuid4().hex[:10])
        self.jobs[job.id] = job
        while len(self.jobs) > MAX_JOBS:
            self.jobs.popitem(last=False)
        threading.Thread(target=self._run, args=(job,), daemon=True).start()
        return job

    def _run(self, job: Job) -> None:
        try:
            with self.gpu:
                job.analyse()
            job.status = "ready"
        except Exception as exc:
            traceback.print_exc()
            job.status, job.error = "error", str(exc)


def _ids(qs: dict, key: str = "ids") -> list[int]:
    raw = qs.get(key, [""])[0]
    return [int(x) for x in re.split(r"[,\s]+", raw) if x]


def make_handler(state: State):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # quieter
            print("[layerize]", fmt % args)

        # -- helpers
        def _send(self, code, body: bytes, ctype="application/json", headers=None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            for k, v in (headers or {}).items():
                self.send_header(k, str(v))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code=200):
            self._send(code, json.dumps(obj).encode())

        def _err(self, code, msg):
            self._json({"error": msg}, code)

        def _authed(self) -> bool:
            return self.headers.get("Authorization", "") == f"Bearer {state.token}"

        def _body(self) -> bytes:
            n = int(self.headers.get("Content-Length", 0))
            if n > MAX_UPLOAD:
                raise ValueError("upload too large")
            return self.rfile.read(n)

        def _job(self, jid: str) -> Job:
            job = state.jobs.get(jid)
            if job is None:
                raise KeyError(f"unknown job {jid}")
            return job

        def _ready(self, jid: str) -> Job:
            job = self._job(jid)
            if job.status != "ready":
                raise RuntimeError(f"job is {job.status}" + (f": {job.error}" if job.error else ""))
            return job

        def _dispatch(self, method: str):
            url = urlparse(self.path)
            qs, path = parse_qs(url.query), url.path.rstrip("/")
            if path == "/healthz":
                return self._json({"ok": True})
            if not self._authed():
                return self._err(401, "bad or missing token")
            try:
                if method == "POST" and path == "/v1/jobs":
                    return self._json({"job_id": state.new_job(self._body()).id}, 202)
                m = re.fullmatch(r"/v1/jobs/(\w+)(?:/(.*))?", path)
                if not m:
                    return self._err(404, "not found")
                jid, sub = m.group(1), m.group(2) or ""
                if method == "GET" and sub == "":
                    job = self._job(jid)
                    body = job.manifest() if job.status == "ready" else {"job_id": jid, "status": job.status}
                    if job.error:
                        body["error"] = job.error
                    return self._json(body)
                job = self._ready(jid)
                if method == "GET" and sub == "sheet.png":
                    ids = _ids(qs) or list(job.regions)[: int(qs.get("limit", ["36"])[0])]
                    with state.gpu:
                        data = job.sheet(ids, cols=int(qs.get("cols", ["6"])[0]))
                    return self._send(200, data, "image/png")
                if method == "POST" and sub == "search":
                    req = json.loads(self._body() or b"{}")
                    with state.gpu:
                        res = job.search(req["query"], int(req.get("top_k", 8)))
                    return self._json({"matches": res})
                if method == "POST" and sub == "segment":
                    req = json.loads(self._body() or b"{}")
                    with state.gpu:
                        reg = job.add_prompt_region(req.get("points"), req.get("box"))
                    return self._json({"region": next(r for r in job.manifest()["regions"] if r["id"] == reg.id)})
                lm = re.fullmatch(r"layers/(\d+)\.png", sub)
                if method == "GET" and lm:
                    with state.gpu:
                        data, lyr = job.png(int(lm.group(1)), qs.get("matte", ["1"])[0] != "0",
                                            qs.get("canvas", ["0"])[0] == "1")
                    canvas = qs.get("canvas", ["0"])[0] == "1"
                    hdr = {"X-Layer-X": 0 if canvas else lyr.x, "X-Layer-Y": 0 if canvas else lyr.y,
                           "X-Layer-W": job.width if canvas else lyr.w, "X-Layer-H": job.height if canvas else lyr.h,
                           "X-Layer-Name": lyr.name}
                    return self._send(200, data, "image/png", hdr)
                if method == "GET" and sub == "background.png":
                    with state.gpu:
                        bg = job.background(_ids(qs, "remove"))
                    buf = io.BytesIO()
                    Image.fromarray(bg).save(buf, "PNG")
                    return self._send(200, buf.getvalue(), "image/png")
                if method == "GET" and sub == "export":
                    with state.gpu:
                        data, name = job.export(_ids(qs), qs.get("fill", ["1"])[0] != "0", qs.get("fmt", ["psd"])[0])
                    return self._send(200, data, "application/octet-stream", {"X-Filename": name})
                return self._err(404, "not found")
            except KeyError as exc:
                return self._err(404, str(exc))
            except (ValueError, RuntimeError, json.JSONDecodeError) as exc:
                return self._err(400, str(exc))
            except Exception as exc:  # pragma: no cover
                traceback.print_exc()
                return self._err(500, f"{type(exc).__name__}: {exc}")

        def do_GET(self):
            self._dispatch("GET")

        def do_POST(self):
            self._dispatch("POST")

    return Handler


def serve(engines, token: str | None = None, host: str = "0.0.0.0", port: int = 8000, block: bool = True):
    token = token or secrets.token_urlsafe(24)
    state = State(engines, token)
    httpd = ThreadingHTTPServer((host, port), make_handler(state))
    if block:
        httpd.serve_forever()
    else:
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, token


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--token", default=None)
    ap.add_argument("--no-warm", action="store_true")
    args = ap.parse_args()
    from .engines import Engines

    eng = Engines()
    if not args.no_warm:
        eng.warm()
    httpd, tok = serve(eng, args.token, port=args.port, block=False)
    print("token:", tok)
    threading.Event().wait()
