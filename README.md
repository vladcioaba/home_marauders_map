# home_marauders_map

A live, multi-camera Marauder's Map for your house. Detects people (and
optionally pets) on every camera, links identities across cameras, and stamps
sepia footprints on a parchment floor plan in real time.

Architecture: one **central command** (Raspberry Pi 5 or a Mac mini) reads
streams from your IP cameras and serves a local **web UI** at
`http://<host>:8000/`. Two pages share the same parchment renderer:

- **/view** — the live Marauder's Map. Watch every footstep in the house.
- **/edit** — sketch the floor plan, drop cameras (RTSP/HTTP URL), set their
  gaze. Save writes back to `config/house.yaml`.

Built Mac-first; Raspberry Pi 5 + Hailo-10H deployment scaffolded for the
detection backend.

## Features

**Detection & tracking**
- Ultralytics YOLO backend — boxes (`yolov8n.pt`) or pose (`yolov8n-pose.pt`)
- Per-camera tracker: ByteTrack (default) or BoT-SORT
- COCO 80 classes — filter with `--classes person,cat,dog,bird` etc.
- Hailo-10H backend scaffolded (inference untested without hardware)

**Multi-camera & geometry**
- Threaded capture — USB webcams, video files (auto-loops), RTSP URLs
- House layout in `config/house.yaml`: rooms, doors, camera placement
- Click-to-calibrate homography per camera (`scripts/calibrate.py`)
- Foot-point projection from bbox → floor-plan meters

**Cross-camera identity**
- Global IDs persist when a person moves between cameras
- Sticky mapping + spatial nearest-neighbor + appearance tiebreaker (HSV histogram ReID)
- Configurable: `--max-distance-m`, `--max-gap-s`, `--trail-s`, `--no-reid`

**Marauder's Map renderer**
- Parchment background, sepia ink rooms / doors / camera icons
- Footprint trails (paired ovals along each path, alternating L/R)
- `--names "Vlad,Hermione,Ron"` overrides the default `P1`, `P2`, `P3`
- `m` toggles **Mischief Managed** (hides people; press again to reveal)
- Footer reads *"I solemnly swear that I am up to no good"* when active

**Persistence & replay**
- SQLite history of positions (`--log tracks.db`), rate-limited per person
- `scripts/replay.py` animates a recorded session — window or MP4 output

## Quickstart (Mac)

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e .

# Web UI — the primary surface.
marauders serve                                    # http://localhost:8000/
marauders serve --live --config config/house.yaml  # also start tracking pipeline
marauders serve --host 0.0.0.0 --port 8000         # expose on the LAN

# Headless / debug viewer (cv2 window, no server). Backwards compat.
marauders                        # webcam, single-cam preview
marauders view --config config/house.yaml --names "Vlad,Hermione"
```

In the web UI: open **/edit** first to sketch rooms (click-drag) and place
cameras (click → enter RTSP URL). Save. Then open **/view** (with
`--live` running) to see footprints walking around the parchment.

### Faster on Mac

```bash
marauders --device mps --imgsz 416     # ~2× faster on Apple Silicon than 640
marauders --map-fps 15                  # cap floor-plan redraws
```

### Log + replay

```bash
marauders --config config/house.yaml --log tracks.db
python scripts/replay.py --db tracks.db --names "Vlad,Hermione" --speed 5
python scripts/replay.py --db tracks.db --out replay.mp4 --fps 15 --speed 10
```

### Calibrate a camera → floor plan

```bash
python scripts/calibrate.py
```

Click alternating image/floor-plan points (≥4 pairs) and press `s` to save
to `config/calibration.yaml`.

First run auto-downloads YOLO weights (~6 MB). macOS will prompt for camera
permission. Press `q` to quit, `m` to toggle the map.

## Flags

```
--config PATH                house.yaml (multi-cam + floor-plan mode)
--source 0|path|rtsp://...   single-cam shortcut
--backend ultralytics|hailo  detector backend
--model  yolov8n.pt          weights (.pt for ultralytics, .hef for hailo)
--tracker bytetrack.yaml     or botsort.yaml
--conf   0.25                detection confidence threshold
--device cpu|mps|cuda        ultralytics-only torch device
--imgsz  640                 inference image size (try 416 / 320 for speed)
--classes person             COCO classes; e.g. 'person,cat,dog'
--names  "Vlad,Hermione"     map global IDs to display names
--map-fps 20                 cap floor-plan redraw FPS
--log PATH                   SQLite history (needs --config)
--max-distance-m 1.5         handoff radius for global ID linking
--max-gap-s 3.0              handoff time tolerance
--trail-s 3.0                footprint trail history
--no-reid                    disable appearance tiebreaker (debug)
```

## Develop

```bash
pip install -e '.[dev]'
pytest                       # unit tests
ruff check src tests         # lint
```

## Deploy to a Raspberry Pi

```bash
PI_HOST=pi@pi5.local ./scripts/deploy_to_pi.sh --setup     # first time
PI_HOST=pi@pi5.local ./scripts/deploy_to_pi.sh             # re-sync + run
PI_HOST=pi@pi5.local BACKEND=hailo ./scripts/deploy_to_pi.sh
```

Env: `PI_HOST` (required), `PI_DIR` (default `~/home_marauders_map`),
`BACKEND`, `ARGS` (extra args passed to `marauders`).
