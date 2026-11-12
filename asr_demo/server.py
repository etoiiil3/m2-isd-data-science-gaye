"""Live ASR demo server. Run it on YOUR laptop (browsers only give microphone access on localhost/HTTPS):

    python -m asr_demo.server            # then open http://localhost:8000

Protocol (one WebSocket per recording, backend chosen per recording):
  client -> {"type":"start","backend":NAME,"reference":optional text}
  client -> binary chunks: float32 mono 16 kHz, ~100 ms each, while the record button is down
  client -> {"type":"stop"}
  server -> {"type":"loading"} (first use of a backend; the page also preloads it via POST /api/load on selection) · {"type":"partial","text","t_ms"} · {"type":"final",...metrics}
Every request is logged to results/results.jsonl (task demo) and its audio saved for replay on other backends.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import time
import traceback
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel

from asr_demo import backends
from asr_demo.metrics import run_session_offline, summarise, wer_or_none
from common.bench import BenchRecord, log_record

ROOT = Path(__file__).resolve().parents[1]
REC = ROOT / "results" / "demo_recordings"
LOG = ROOT / "results" / "results.jsonl"
app = FastAPI(title="ASR live demo")


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/api/backends")
def list_backends():
    return backends.available()


def _log(m: dict, mode: str):
    cfg = {"mode": mode, "streaming": m.get("streaming"), "recording": m.get("recording")}
    keep = {k: v for k, v in m.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    log_record(BenchRecord("demo", m["backend"], cfg, keep), LOG)


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    name = None
    try:
        start = json.loads(await ws.receive_text())
        name, reference = start["backend"], start.get("reference") or ""
        if name not in backends.REGISTRY:
            await ws.send_json({"type": "error", "message": f"unknown backend {name}"})
            return
        if not backends.available()[[b["name"] for b in backends.available()].index(name)]["loaded"]:
            await ws.send_json({"type": "loading"})
        t_load = time.perf_counter()
        try:
            backend = await asyncio.to_thread(backends.get, name)
        except backends.BackendUnavailable as e:
            await ws.send_json({"type": "error", "message": str(e)})
            return
        load_ms = (time.perf_counter() - t_load) * 1000
        session = backend.new_session()
        audio, compute, t0, first_partial, n_partials = [], 0.0, None, None, 0
        while True:
            msg = await ws.receive()
            if msg.get("bytes") is not None:
                chunk = np.frombuffer(msg["bytes"], dtype=np.float32).copy()
                t0 = t0 or time.perf_counter()
                audio.append(chunk)
                c0 = time.perf_counter()
                text = await asyncio.to_thread(session.accept_chunk, chunk)
                compute += time.perf_counter() - c0
                if text:
                    n_partials += 1
                    first_partial = first_partial or (time.perf_counter() - t0) * 1000
                    await ws.send_json({"type": "partial", "text": text, "t_ms": (time.perf_counter() - t0) * 1000})
            elif msg.get("text") is not None and json.loads(msg["text"]).get("type") == "stop":
                t_stop = time.perf_counter()
                final = await asyncio.to_thread(session.finish)
                done = time.perf_counter()
                compute += done - t_stop
                y = np.concatenate(audio) if audio else np.zeros(0, np.float32)
                rid = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
                REC.mkdir(parents=True, exist_ok=True)
                sf.write(REC / f"{rid}.wav", y, 16000)
                if reference:
                    (REC / f"{rid}.txt").write_text(reference)
                m = summarise(name, backend.streaming, y, 16000, compute, first_partial, (done - t_stop) * 1000,
                              n_partials, final, reference, load_ms)
                m["recording"] = rid
                _log(m, "live")
                await ws.send_json({"type": "final", **m})
                return
            elif msg.get("type") == "websocket.disconnect":
                return
    except WebSocketDisconnect:
        return
    except Exception as e:  # tell the browser why, instead of silently dropping the socket
        traceback.print_exc()
        with contextlib.suppress(Exception):
            await ws.send_json({"type": "error", "message": f"{name}: {backends.explain(e)}" if name else backends.explain(e)})
    finally:  # a proper close: returning without one drops the TCP connection, and the browser may lose our last message
        with contextlib.suppress(Exception):
            await ws.close()


class LoadRequest(BaseModel):
    backend: str


@app.post("/api/load")
async def load(req: LoadRequest):
    """Load a backend as soon as it is selected, so the first recording does not wait (and a stub is reported now)."""
    try:
        await asyncio.to_thread(backends.get, req.backend)
    except backends.BackendUnavailable as e:
        raise HTTPException(status_code=409, detail=str(e))
    return {"loaded": True}


@app.get("/api/prompt")
def prompt(backend: str):
    """A sentence suited to this backend (digits for the day-1 entries), or null: the page then uses its own list."""
    return {"text": backends.prompt(backend)}


class ReplayRequest(BaseModel):
    recording: str
    backend: str
    realtime: bool = False


@app.post("/api/replay")
async def replay(req: ReplayRequest):
    """Same saved audio, another backend: the only fair way to compare backends qualitatively."""
    y, sr = sf.read(REC / f"{req.recording}.wav", dtype="float32")
    ref_f = REC / f"{req.recording}.txt"
    reference = ref_f.read_text() if ref_f.exists() else ""
    try:
        backend = await asyncio.to_thread(backends.get, req.backend)
    except backends.BackendUnavailable as e:
        raise HTTPException(status_code=409, detail=str(e))
    m = await asyncio.to_thread(run_session_offline, backend, y, sr, reference, req.realtime)
    m["recording"] = req.recording
    _log(m, "replay")
    return m


def main():
    import argparse
    import importlib.util
    import uvicorn
    if not any(importlib.util.find_spec(m) for m in ("websockets", "wsproto")):
        raise SystemExit("no WebSocket library: live recording (/ws) cannot work. "
                         "Run: pip install 'uvicorn[standard]'")
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    uvicorn.run(app, host=a.host, port=a.port)


if __name__ == "__main__":
    main()
