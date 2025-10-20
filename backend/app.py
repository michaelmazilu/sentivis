from __future__ import annotations

import logging
import time
from typing import Any, Dict

from flask import Flask, jsonify, request
from flask_socketio import SocketIO, emit

from sentivis import InferencePipeline

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("sentivis.backend")

app = Flask(__name__)
app.config["SECRET_KEY"] = "sentivis-dev-secret"

socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode="eventlet",
    max_http_buffer_size=6 * 1024 * 1024,
)

pipeline = InferencePipeline()


@app.get("/healthz")
def healthz() -> Any:
    return jsonify({"status": "ok", "client": request.remote_addr})


@socketio.on("connect")
def handle_connect() -> None:
    LOGGER.info("Client connected: %s", request.sid)
    emit("server_status", {"status": "ready"})


@socketio.on("disconnect")
def handle_disconnect() -> None:
    LOGGER.info("Client disconnected: %s", request.sid)
    pipeline.reset()


@socketio.on("frame")
def handle_frame(data: Dict[str, Any]) -> None:
    if not isinstance(data, dict):
        emit("error", {"message": "frame payload must be a dictionary"})
        return

    frame_payload = data.get("frame")
    frame_id = data.get("frameId")
    client_ts = data.get("timestamp")

    start_time = time.time()

    try:
        result = pipeline.process_frame(frame_payload)
    except ValueError as exc:
        LOGGER.warning("Frame rejected: %s", exc)
        emit("error", {"message": str(exc), "frameId": frame_id})
        return
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.exception("Unexpected failure while processing frame")
        emit("error", {"message": "internal-error", "frameId": frame_id})
        raise exc

    latency_ms = (time.time() - start_time) * 1000.0

    result.update(
        {
            "frameId": frame_id,
            "metrics": {
                "latencyMs": latency_ms,
                "clientTimestamp": client_ts,
            },
        }
    )

    emit("inference", result)


@socketio.on_error()  # type: ignore
def error_handler(err: Exception) -> None:
    LOGGER.exception("Socket error: {0}", err)


def main() -> None:
    LOGGER.info("Starting Sentivis backend...")
    socketio.run(app, host="0.0.0.0", port=5001)


if __name__ == "__main__":
    main()
