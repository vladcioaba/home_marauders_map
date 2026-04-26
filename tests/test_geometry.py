from __future__ import annotations

import numpy as np
import pytest

from marauders_map.geometry import compute_homography, foot_point, project_point


def test_foot_point_is_bottom_center():
    assert foot_point((10.0, 20.0, 30.0, 50.0)) == (20.0, 50.0)


def test_compute_homography_identity_like():
    # Map four image-pixel points to identical floor-meter points → identity.
    pts = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)
    H = compute_homography(pts, pts)
    assert H is not None
    # project_point should round-trip (within float tolerance).
    for p in pts:
        x, y = project_point(H, (float(p[0]), float(p[1])))
        assert x == pytest.approx(p[0], abs=1e-3)
        assert y == pytest.approx(p[1], abs=1e-3)


def test_compute_homography_scales_floor_plane():
    # 1 pixel = 0.1 meter scaling.
    img = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
    flr = img * 0.1
    H = compute_homography(img, flr)
    assert H is not None
    x, y = project_point(H, (50.0, 50.0))
    assert x == pytest.approx(5.0, abs=1e-3)
    assert y == pytest.approx(5.0, abs=1e-3)


def test_compute_homography_too_few_points():
    pts = np.array([[0, 0], [1, 0], [1, 1]], dtype=np.float32)
    assert compute_homography(pts, pts) is None


def test_compute_homography_mismatched_shapes():
    a = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
    b = np.array([[0, 0], [1, 0], [1, 1]], dtype=np.float32)
    assert compute_homography(a, b) is None
