"""Interactive per-camera homography calibration.

For each camera in the house config:
  1. Grab one frame.
  2. Click matching points, alternating: camera image → floor plan image.
  3. Need at least 4 matched pairs on non-collinear floor points.
  4. Keys: `s` save this camera, `u` undo last click, `n` skip, `q` quit all.

Output: config/calibration.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from marauders_map.capture import ThreadedCapture  # noqa: E402
from marauders_map.house import CameraConfig, House  # noqa: E402
from marauders_map.viz import draw_floorplan  # noqa: E402

QUIT = object()


def _render(
    frame: np.ndarray, img_pts: list[list[float]],
    plan: np.ndarray, flr_pts: list[list[float]], scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    f = frame.copy()
    for i, (x, y) in enumerate(img_pts):
        cv2.circle(f, (int(x), int(y)), 6, (0, 255, 0), -1)
        cv2.putText(f, str(i + 1), (int(x) + 8, int(y) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    p = plan.copy()
    for i, (mx, my) in enumerate(flr_pts):
        px, py = int(mx * scale), int(my * scale)
        cv2.circle(p, (px, py), 6, (0, 255, 0), -1)
        cv2.putText(p, str(i + 1), (px + 8, py - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return f, p


def _calibrate_one(
    cam: CameraConfig, frame: np.ndarray, plan: np.ndarray, scale: float,
) -> tuple[list, list] | object | None:
    """Returns (img_pts, flr_pts), None (skip), or QUIT sentinel."""
    img_pts: list[list[float]] = []
    flr_pts: list[list[float]] = []
    expecting = ["image"]  # list so inner funcs can mutate

    cam_win = f"{cam.id} — camera (click floor features)"
    plan_win = f"{cam.id} — floor plan (click matching points)"

    def refresh() -> None:
        f, p = _render(frame, img_pts, plan, flr_pts, scale)
        cv2.imshow(cam_win, f)
        cv2.imshow(plan_win, p)

    def on_image(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and expecting[0] == "image":
            img_pts.append([float(x), float(y)])
            expecting[0] = "floor"
            refresh()

    def on_plan(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and expecting[0] == "floor":
            flr_pts.append([x / scale, y / scale])
            expecting[0] = "image"
            refresh()

    cv2.namedWindow(cam_win)
    cv2.namedWindow(plan_win)
    cv2.setMouseCallback(cam_win, on_image)
    cv2.setMouseCallback(plan_win, on_plan)
    refresh()

    try:
        while True:
            key = cv2.waitKey(20) & 0xFF
            if key == ord("s"):
                if len(img_pts) == len(flr_pts) >= 4:
                    return img_pts, flr_pts
                print(f"  need ≥4 matched pairs, have "
                      f"{len(img_pts)} image / {len(flr_pts)} floor")
            elif key == ord("u"):
                if expecting[0] == "floor" and img_pts:
                    img_pts.pop()
                    expecting[0] = "image"
                elif expecting[0] == "image" and flr_pts:
                    flr_pts.pop()
                    expecting[0] = "floor"
                refresh()
            elif key == ord("n"):
                return None
            elif key == ord("q"):
                return QUIT
    finally:
        cv2.destroyWindow(cam_win)
        cv2.destroyWindow(plan_win)


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive camera calibration")
    parser.add_argument("--config", type=Path, default=Path("config/house.yaml"))
    parser.add_argument("--out", type=Path, default=Path("config/calibration.yaml"))
    args = parser.parse_args()

    house = House.load(args.config)
    plan = draw_floorplan(house)
    scale = house.floorplan.scale

    existing: dict[str, dict] = {}
    if args.out.exists():
        existing = (yaml.safe_load(args.out.read_text()) or {}).get("cameras") or {}

    print("Controls: click alternating — camera → floor plan.")
    print("  s = save this camera   u = undo last   n = skip   q = quit")

    for cam in house.cameras:
        print(f"\n[{cam.id}] {cam.name} — source {cam.source}")
        cap = ThreadedCapture(cam.source).start()
        frame = None
        for _ in range(60):
            frame = cap.latest()
            if frame is not None:
                break
            cv2.waitKey(50)
        cap.stop()
        if frame is None:
            print(f"  skipped — no frame from {cam.source}")
            continue

        result = _calibrate_one(cam, frame, plan, scale)
        if result is QUIT:
            print("  quit requested")
            break
        if result is None:
            print("  skipped")
            continue
        img_pts, flr_pts = result  # type: ignore[misc]
        existing[cam.id] = {"image": img_pts, "floor": flr_pts}
        print(f"  saved {len(img_pts)} point pairs for {cam.id}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(yaml.safe_dump({"cameras": existing}, sort_keys=False))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
