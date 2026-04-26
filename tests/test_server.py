from __future__ import annotations

from pathlib import Path

import pytest

from marauders_map.house import House
from marauders_map.server import create_app


@pytest.fixture
def client(tmp_path: Path):
    app = create_app(tmp_path / "house.yaml")
    app.config["TESTING"] = True
    return app.test_client(), tmp_path / "house.yaml"


def test_index_serves_landing(client):
    c, _ = client
    res = c.get("/")
    assert res.status_code == 200
    body = res.data.decode()
    assert "Marauder" in body
    assert "View" in body
    assert "Edit" in body


def test_view_page_loads(client):
    c, _ = client
    res = c.get("/view")
    assert res.status_code == 200
    assert b"plan" in res.data


def test_edit_page_loads(client):
    c, _ = client
    res = c.get("/edit")
    assert res.status_code == 200
    assert b"toolbar" in res.data


def test_get_house_returns_default_when_missing(client):
    c, _ = client
    res = c.get("/api/house")
    assert res.status_code == 200
    data = res.get_json()
    assert "floorplan" in data
    assert data["rooms"] == []
    assert data["cameras"] == []


def test_post_house_persists_yaml(client):
    c, path = client
    payload = {
        "floorplan": {"size": [8.0, 5.0], "scale": 60.0},
        "rooms": [{"id": "r1", "name": "R1", "rect": [0.0, 0.0, 4.0, 3.0]}],
        "doors": [],
        "cameras": [{
            "id": "cam0", "name": "cam0", "source": "rtsp://1.2.3.4/stream",
            "position": [4.0, 2.5], "heading": 0.0, "fov": 90.0,
        }],
    }
    res = c.post("/api/house", json=payload)
    assert res.status_code == 200
    assert res.get_json()["ok"] is True

    loaded = House.load(path)
    assert len(loaded.rooms) == 1
    assert loaded.cameras[0].source == "rtsp://1.2.3.4/stream"


def test_post_house_rejects_non_object(client):
    c, _ = client
    res = c.post("/api/house", json=["nope"])
    assert res.status_code == 400


def test_state_default_is_idle(client):
    c, _ = client
    res = c.get("/api/state")
    data = res.get_json()
    assert data["markers"] == []
    assert data["trails"] == {}
    assert data["running"] is False


def test_add_camera_appends_to_yaml(client):
    c, path = client
    res = c.post("/api/cameras", json={"source": "rtsp://10.0.0.5/stream"})
    assert res.status_code == 200
    body = res.get_json()
    assert body["ok"] is True
    assert body["id"] == "cam1"  # first camera in an empty file

    loaded = House.load(path)
    assert len(loaded.cameras) == 1
    assert loaded.cameras[0].source == "rtsp://10.0.0.5/stream"


def test_add_camera_requires_source(client):
    c, _ = client
    res = c.post("/api/cameras", json={"name": "no source"})
    assert res.status_code == 400


def test_static_assets_served(client):
    c, _ = client
    res = c.get("/static/styles.css")
    assert res.status_code == 200
    assert b"--parchment" in res.data
    res = c.get("/static/common.js")
    assert res.status_code == 200
    assert b"renderHouse" in res.data


def test_stream_returns_multipart_with_placeholder(client):
    """Without --live, the route still emits one placeholder JPEG and closes."""
    c, _ = client
    res = c.get("/stream/cam0.mjpg")
    assert res.status_code == 200
    assert res.mimetype == "multipart/x-mixed-replace"
    # Read enough of the stream to see the first JPEG frame and its boundary.
    body = b""
    for chunk in res.response:
        body += chunk
        if b"--frame" in body and b"\xff\xd8" in body:  # SOI marker
            break
        if len(body) > 200_000:
            break
    res.close()
    assert b"image/jpeg" in body
    assert b"\xff\xd8" in body  # JPEG start-of-image
    assert b"\xff\xd9" in body  # JPEG end-of-image


def test_stream_after_live_state_set(client):
    """When LiveState has a frame, the stream serves it (not the placeholder)."""
    import numpy as np
    c, _ = client
    state = c.application.config["LIVE_STATE"]
    fake = np.full((48, 64, 3), 200, dtype=np.uint8)
    state.set_frame("cam0", fake)

    assert state.latest_frame("cam0") is not None
    res = c.get("/stream/cam0.mjpg")
    body = b""
    for chunk in res.response:
        body += chunk
        if len(body) > 50_000:
            break
    res.close()
    assert b"\xff\xd8" in body
