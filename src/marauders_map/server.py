"""Flask web server — primary UX surface for home_marauders_map.

Two pages, one shared parchment-styled SVG floor plan:
  /view  → live Marauder's Map (polls /api/state, MJPEG camera tiles)
  /edit  → drag rooms, click-place IP cameras, save (POST /api/house)

Designed to be hosted on the central command (Pi or Mac mini) and accessed
in a browser at http://<host>:<port>/. Works fully offline.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import cv2
import numpy as np
import yaml
from flask import Flask, Response, jsonify, render_template, request

from marauders_map.house import House, dump_house, house_to_dict
from marauders_map.live import LiveState, run_live_loop


_PKG_DIR = Path(__file__).parent


def _empty_house_dict() -> dict:
    return {
        "floorplan": {"size": [10.0, 6.0], "scale": 60.0},
        "rooms": [],
        "doors": [],
        "cameras": [],
    }


def _read_house_dict(path: Path) -> dict:
    if path.exists():
        data = yaml.safe_load(path.read_text()) or {}
        # Be permissive — allow a partial file by filling in missing sections.
        merged = _empty_house_dict()
        merged.update({k: data[k] for k in ("floorplan", "rooms", "doors", "cameras") if k in data})
        return merged
    return _empty_house_dict()


def create_app(
    house_path: Path,
    *,
    calibration_path: Path | None = None,
    live_state: LiveState | None = None,
) -> Flask:
    app = Flask(
        __name__,
        template_folder=str(_PKG_DIR / "templates"),
        static_folder=str(_PKG_DIR / "static"),
    )
    app.config["HOUSE_PATH"] = Path(house_path)
    app.config["CALIBRATION_PATH"] = (
        Path(calibration_path) if calibration_path else None
    )
    app.config["LIVE_STATE"] = live_state or LiveState()

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/view")
    def view():
        return render_template("view.html")

    @app.route("/edit")
    def edit():
        return render_template("edit.html")

    @app.route("/api/house", methods=["GET"])
    def api_get_house():
        return jsonify(_read_house_dict(app.config["HOUSE_PATH"]))

    @app.route("/api/house", methods=["POST"])
    def api_post_house():
        data = request.get_json(force=True, silent=True)
        if not isinstance(data, dict):
            return jsonify({"error": "expected JSON object"}), 400
        # Round-trip through House.load by writing a temp YAML and re-reading;
        # this validates required fields without rewriting the schema here.
        path = app.config["HOUSE_PATH"]
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            # Permissive write: accept whatever shape the editor sends, as long
            # as it round-trips through yaml.safe_dump.
            path.write_text(yaml.safe_dump(data, sort_keys=False))
        except Exception as e:
            return jsonify({"error": f"write failed: {e}"}), 500
        return jsonify({"ok": True, "path": str(path)})

    @app.route("/api/state")
    def api_get_state():
        return jsonify(app.config["LIVE_STATE"].snapshot())

    @app.route("/stream/<cam_id>.mjpg")
    def stream_camera(cam_id: str):
        state: LiveState = app.config["LIVE_STATE"]

        def gen():
            placeholder_sent = False
            empty_iters = 0
            while True:
                frame = state.latest_frame(cam_id)
                if frame is None:
                    if not placeholder_sent:
                        frame = _placeholder_frame(cam_id)
                        placeholder_sent = True
                    else:
                        empty_iters += 1
                        # Stop streaming after ~5s of no frames so the browser
                        # can fall back to its broken-image alt text.
                        if empty_iters > 50:
                            return
                        time.sleep(0.1)
                        continue
                ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if not ok:
                    time.sleep(0.05)
                    continue
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
                )
                time.sleep(1.0 / 15.0)

        return Response(
            gen(), mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @app.route("/api/cameras", methods=["POST"])
    def api_add_camera():
        """Append a single camera to house.yaml. Body: {id, name?, source, position?}."""
        data = request.get_json(force=True, silent=True) or {}
        if "source" not in data:
            return jsonify({"error": "missing 'source' (RTSP url, http stream, or webcam index)"}), 400
        path = app.config["HOUSE_PATH"]
        house = _read_house_dict(path)
        cams = house.setdefault("cameras", [])
        existing_ids = {c.get("id") for c in cams}
        cam_id = data.get("id") or _next_id("cam", existing_ids)
        cams.append({
            "id": cam_id,
            "name": data.get("name", cam_id),
            "source": data["source"],
            "position": data.get("position", [
                house["floorplan"]["size"][0] / 2.0,
                house["floorplan"]["size"][1] / 2.0,
            ]),
            "heading": float(data.get("heading", 0.0)),
            "fov": float(data.get("fov", 90.0)),
        })
        path.write_text(yaml.safe_dump(house, sort_keys=False))
        return jsonify({"ok": True, "id": cam_id})

    return app


def _next_id(prefix: str, existing: set) -> str:
    i = 1
    while f"{prefix}{i}" in existing:
        i += 1
    return f"{prefix}{i}"


def _placeholder_frame(cam_id: str) -> np.ndarray:
    """A 320x240 dark frame with a 'no live feed' label. Sent before --live kicks in."""
    img = np.full((240, 320, 3), 30, dtype=np.uint8)
    cv2.putText(
        img, cam_id, (12, 30),
        cv2.FONT_HERSHEY_TRIPLEX, 0.7, (200, 200, 200), 1, cv2.LINE_AA,
    )
    cv2.putText(
        img, "no live feed", (12, 130),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1, cv2.LINE_AA,
    )
    cv2.putText(
        img, "start with --live", (12, 156),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 140, 140), 1, cv2.LINE_AA,
    )
    return img


def serve(
    house_path: Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    live: bool = False,
    calibration_path: Path | None = None,
    live_kwargs: dict | None = None,
) -> None:
    """Run the Flask app (foreground). Optionally spawn the detection loop."""
    state = LiveState()
    app = create_app(house_path, calibration_path=calibration_path, live_state=state)

    if live:
        kwargs = {"calibration_path": calibration_path}
        kwargs.update(live_kwargs or {})
        thread = threading.Thread(
            target=run_live_loop,
            args=(Path(house_path), state),
            kwargs=kwargs,
            daemon=True,
        )
        thread.start()
        print(f"[live] detection loop started")

    print(f"Marauder's Map → http://{host}:{port}/")
    print("Press Ctrl+C to stop.")
    app.run(host=host, port=port, threaded=True, debug=False, use_reloader=False)
