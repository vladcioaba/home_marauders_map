# home_marauders_map

Real-time object detection + tracking on a live camera feed, inspired by the
Hailo / Ultralytics YOLO26-on-Raspberry-Pi-5 demo (90 FPS small / 48 FPS medium
on Hailo-10H AI HAT+2).

## Goal

Track a person moving through a house across **multiple cameras**, showing
their live position on a **floor plan**. Detection is per-frame, tracking is
per-camera, and identities are linked across cameras so one person keeps a
single global ID from room to room.

- **Detector:** Ultralytics YOLO (YOLO26 on Hailo target; YOLOv8/11 on Mac).
- **Per-camera tracker:** ByteTrack or BoT-SORT (built into Ultralytics).
- **Geometry:** each camera calibrated with a homography from its ground plane
  to the shared floor-plan coordinate system.
- **Cross-camera linking:** handoff via door/region constraints + time, with an
  optional appearance/ReID model for robustness when people look different.
- **Output:** per-camera annotated frames **plus** a live top-down floor-plan
  view with each person's position and a short trail.

## Architecture (target)

```
  [cam 0]──▶ detector ──▶ ByteTrack ──┐
  [cam 1]──▶ detector ──▶ ByteTrack ──┼──▶ global tracker ──▶ floor-plan view
  [cam N]──▶ detector ──▶ ByteTrack ──┘         │
                                                ▼
                                    per-camera homography
                                    + room/door graph
```

Key pieces:

1. **Capture pool** — one thread/process per stream (USB / RTSP / file).
2. **Per-camera pipeline** — detector → tracker → foot-point in image.
3. **Homography** — projects each track's foot point onto the floor plan.
4. **Global tracker** — maintains global IDs by fusing local tracks across
   cameras using proximity on the floor plan, time, and (optionally) appearance.
5. **Floor-plan renderer** — draws rooms, camera positions/FOV, and live dots
   with trails.

## Hardware plan

| Phase | Machine | Runtime |
|-------|---------|---------|
| Dev (now) | MacBook (Apple Silicon, macOS) | PyTorch / CoreML / MPS |
| Deploy (later) | Raspberry Pi 5 + Hailo-10H "AI HAT+2" | HailoRT + compiled `.hef` |

Code must run on the Mac without Hailo hardware. Keep the inference backend
behind a small interface so we can swap in HailoRT later.

## Repo layout (target)

```
home_marauders_map/
├── CLAUDE.md                # this file
├── README.md                # user-facing quickstart
├── pyproject.toml           # deps + tooling
├── config/
│   ├── house.yaml           # cameras list, floor plan, rooms, doors
│   └── floorplan.png        # top-down image of the house
├── src/
│   └── marauders_map/
│       ├── __init__.py
│       ├── app.py           # multi-cam entrypoint (single-cam shortcut via --source)
│       ├── capture.py       # stream sources (threaded)
│       ├── detector.py      # backend-agnostic interface
│       ├── backends/
│       │   ├── ultralytics_backend.py   # Mac / CPU / MPS
│       │   └── hailo_backend.py         # Pi + Hailo (later)
│       ├── iou_tracker.py   # standalone IoU tracker for non-Ultralytics backends
│       ├── geometry.py      # homography, foot-point projection
│       ├── house.py         # floor plan, rooms, cameras, door graph
│       ├── tracker_global.py# cross-camera ID linker
│       ├── reid.py          # appearance embeddings (HSV histogram)
│       ├── viz.py           # Marauder's Map renderer (parchment + footprints)
│       ├── classes.py       # COCO class table + resolver
│       └── logger.py        # SQLite event log
├── scripts/
│   ├── run_mac.sh
│   └── deploy_to_pi.sh      # rsync + remote run
└── tests/
```

## Dev environment (Mac)

- Python 3.11 (matches current Ultralytics + torch wheels).
- `uv` for env + deps (fast, single-file lock). `pip`/`venv` works too.
- Webcam access requires macOS camera permission for the terminal / IDE.

Minimal deps:

- `ultralytics` — YOLO models + trackers
- `opencv-python` — capture, draw, display
- `numpy`
- `torch` (installed via Ultralytics; uses `mps` on Apple Silicon)

## How to resume a session

When starting a fresh Claude Code session in this directory:

1. Claude auto-loads this `CLAUDE.md`.
2. Tell Claude what you want to work on next (see **Status / next steps**).
3. If something here is stale, update it before continuing.

## Phased plan

**Phase 1 — single-camera PoC (done)**
- [x] Scaffold `pyproject.toml`, `src/marauders_map/`, minimal `app.py`.
- [x] Mac path: open webcam with OpenCV, run YOLOv8n + ByteTrack, render boxes.
- [x] CLI flags: `--source`, `--model`, `--tracker`, `--conf`, `--device`.
- [ ] First-run verification on user's Mac (camera permission, FPS, quit with `q`).
- [ ] Measure Mac baseline FPS (MPS vs CPU).

**Phase 2 — multi-stream**
- [x] `capture.py`: threaded reader per stream.
- [x] `config/house.yaml` + `house.py` loader (cameras only for now).
- [x] `app.py` supports `--config` multi-cam mode with tiled display and a
      per-camera YOLO instance (so tracker state stays separate).
- [ ] First run with 2+ sources (webcam + a clip in `samples/`).

**Phase 3 — floor plan + geometry**
- [x] `house.yaml`: rooms (rects), doors (segments), camera placement.
- [x] `house.py`: `Floorplan`, `Room`, `Door`, `CameraConfig`, `Calibration`,
      `House.load()` with optional `config/calibration.yaml`.
- [x] `geometry.py`: `compute_homography`, `project_point`, `foot_point`.
- [x] `viz.py`: `draw_floorplan`, `draw_markers` with color per track ID.
- [x] `scripts/calibrate.py`: alternating-click tool, writes
      `config/calibration.yaml` (separate from hand-authored `house.yaml`).
- [x] `app.py`: person-class filter, floor-plan window with live dots.
- [ ] First real run: sketch a rough floor plan of the current room, run
      calibrate.py on the Mac webcam, verify dots land in the right rectangle.
- [ ] Add a short position history / trail overlay (2–3 s).

**Phase 4 — cross-camera identity**
- [x] `tracker_global.py`: `GlobalTracker` with sticky per-camera mapping +
      greedy nearest-neighbor on the floor plan (`max_distance_m`, `max_gap_s`,
      `trail_s`, `expire_s`).
- [x] `house.py`: `room_at()` point-in-rectangle lookup.
- [x] `viz.py`: `draw_trails` + room label in marker ("P1 @ kitchen").
      Color is keyed off global ID so it stays stable across cameras.
- [x] `app.py`: overlays `P<gid>` on each camera's annotated frame so the
      camera view and floor-plan view agree on identity.
- [ ] Tune `max_distance_m` / `max_gap_s` once first real runs are in.
- [x] `reid.py`: HSV histogram embedding (no extra deps), used as an
      appearance tiebreaker in `GlobalTracker._nearest` when multiple
      spatial candidates match. Verified against a crossing scenario.

**Phase 4¾ — class filter (people + pets)**
- [x] `classes.py`: COCO name↔id table and `resolve_classes()`.
- [x] `--classes` CLI: default "person"; pass e.g. "person,cat,dog,bird"
      to also track pets. YOLO already detects all 80 COCO classes.
- [x] Per-detection class carried through `_Detection` → `_overlay_global_ids`
      so each camera bbox label reads `P1 (person)`, `P2 (cat)`, etc.

**Phase 4½ — persistence**
- [x] `logger.py`: SQLite writer (`events` table keyed by time + global_id),
      per-person rate limit (1 row / s default) so the DB stays small.
- [x] `app.py`: `--log PATH` opt-in; flushes per frame under WAL.
- [x] `scripts/replay.py`: reads the DB and animates trails on the floor plan,
      either to a window or to MP4 via `--out`. `--speed`/`--fps` controls
      compression; `--from`/`--to` filters a time range.

**Phase 5 — deploy to Pi + Hailo**
- [x] `detector.py`: `Detector` Protocol + `Detection`, `TrackedFrame`,
      `make_detector(backend, ...)` factory.
- [x] `backends/ultralytics_backend.py`: current YOLO+ByteTrack logic behind
      the interface + histogram embedding per detection.
- [x] `iou_tracker.py`: standalone IoU-based tracker (for backends that don't
      ship their own).
- [x] `backends/hailo_backend.py`: scaffolded with realistic HailoRT API
      shape (letterbox preproc, YOLO head decoder for the two common output
      layouts, IoUTracker, TrackedFrame assembly). **Marked untested** —
      I don't have Hailo hardware to verify the inference bindings.
- [x] `app.py`: `--backend {ultralytics,hailo}` CLI flag.
- [x] `scripts/deploy_to_pi.sh`: rsync + remote venv + remote run.
      Env: `PI_HOST`, `PI_DIR`, `BACKEND`, `ARGS`. `--setup` for first run.
- [ ] On a physical Pi: compile YOLO26 via Hailo Model Zoo, confirm layer
      names in `hailo_backend.py`, measure FPS. Target from the LinkedIn
      post: ~90 FPS (small) / ~48 FPS (medium).

## References

- Hailo post that kicked this off:
  `linkedin.com/posts/hailo-ai_ultralytics-latest-model-yolo26-...`
- Ultralytics docs: https://docs.ultralytics.com/
- Ultralytics trackers: https://docs.ultralytics.com/modes/track/
- HailoRT / Model Zoo: https://github.com/hailo-ai/hailo_model_zoo
