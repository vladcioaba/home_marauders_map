#!/usr/bin/env python3
"""Replay recorded person positions on the floor plan.

Reads events from a TrackLogger SQLite database (`tracks.db`) and animates
each person's dot + trail over time. Either displays live (cv2.imshow) or
writes an MP4 via --out.

Usage:
    python scripts/replay.py --db tracks.db [--config config/house.yaml]
                             [--speed 1.0] [--fps 10]
                             [--from UNIX_TS] [--to UNIX_TS]
                             [--out replay.mp4]
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from marauders_map.house import House  # noqa: E402
from marauders_map.tracker_global import GlobalMarker  # noqa: E402
from marauders_map.viz import (  # noqa: E402
    draw_floorplan, draw_markers, draw_trails, headings_from_trails,
)


def _load_events(
    db: Path, t_from: float | None, t_to: float | None,
) -> list[tuple[float, int, str, float, float]]:
    q = "SELECT t_s, global_id, cam_id, x_m, y_m FROM events"
    params: list[float] = []
    clauses: list[str] = []
    if t_from is not None:
        clauses.append("t_s >= ?"); params.append(t_from)
    if t_to is not None:
        clauses.append("t_s <= ?"); params.append(t_to)
    if clauses:
        q += " WHERE " + " AND ".join(clauses)
    q += " ORDER BY t_s"
    with sqlite3.connect(str(db)) as conn:
        return list(conn.execute(q, params))


def _state_at(
    t_now: float,
    by_gid: dict[int, list[tuple[float, float, float, str]]],
    ts_by_gid: dict[int, list[float]],
    stale_s: float,
    trail_s: float,
) -> tuple[list[GlobalMarker], dict[int, list[tuple[float, float]]]]:
    markers: list[GlobalMarker] = []
    trails: dict[int, list[tuple[float, float]]] = {}
    for gid, arr in by_gid.items():
        ts = ts_by_gid[gid]
        idx = bisect_right(ts, t_now) - 1
        if idx < 0:
            continue
        t_latest, x, y, cam = arr[idx]
        if t_now - t_latest > stale_s:
            continue
        markers.append(GlobalMarker(gid, cam, 0, (x, y)))
        from_i = bisect_right(ts, t_now - trail_s)
        trail = [(arr[i][1], arr[i][2]) for i in range(from_i, idx + 1)]
        if len(trail) >= 2:
            trails[gid] = trail
    return markers, trails


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay recorded positions on the floor plan")
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config/house.yaml"))
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Sim-time compression: 1=realtime, 10=10x faster")
    parser.add_argument("--fps", type=float, default=10.0, help="Render / output FPS")
    parser.add_argument("--trail-s", type=float, default=3.0)
    parser.add_argument("--stale-s", type=float, default=3.0,
                        help="Hide a dot whose latest event is older than this")
    parser.add_argument("--from", dest="t_from", type=float, default=None)
    parser.add_argument("--to", dest="t_to", type=float, default=None)
    parser.add_argument("--out", type=Path, default=None,
                        help="Write MP4 here instead of showing a window")
    parser.add_argument("--names", default=None,
                        help="Comma-separated names mapped by global id (P1, P2, ...)")
    args = parser.parse_args()
    names: dict[int, str] = {}
    if args.names:
        for i, n in enumerate(args.names.split(","), start=1):
            n = n.strip()
            if n:
                names[i] = n

    house = House.load(args.config)
    events = _load_events(args.db, args.t_from, args.t_to)
    if not events:
        print("No events in the selected range.", file=sys.stderr)
        sys.exit(1)

    by_gid: dict[int, list[tuple[float, float, float, str]]] = defaultdict(list)
    for t_s, gid, cam, x, y in events:
        by_gid[gid].append((t_s, x, y, cam))
    ts_by_gid = {gid: [e[0] for e in arr] for gid, arr in by_gid.items()}

    t_start = events[0][0]
    t_end = events[-1][0]
    step = args.speed / args.fps

    writer: cv2.VideoWriter | None = None
    if args.out:
        plan0 = draw_floorplan(house)
        h, w = plan0.shape[:2]
        writer = cv2.VideoWriter(
            str(args.out), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h),
        )
        if not writer.isOpened():
            print(f"Could not open writer for {args.out}", file=sys.stderr)
            sys.exit(1)

    t = t_start
    frames_written = 0
    try:
        while t <= t_end:
            markers, trails = _state_at(
                t, by_gid, ts_by_gid, args.stale_s, args.trail_s,
            )
            img = draw_floorplan(house)
            draw_trails(img, house, trails)
            draw_markers(
                img, house, markers, names=names,
                headings=headings_from_trails(trails),
            )
            cv2.putText(
                img, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t)),
                (10, 22),
                cv2.FONT_HERSHEY_TRIPLEX, 0.5, (40, 70, 110), 1,
            )

            if writer is not None:
                writer.write(img)
            else:
                cv2.imshow("replay", img)
                delay_ms = max(1, int(1000.0 / args.fps))
                if cv2.waitKey(delay_ms) & 0xFF == ord("q"):
                    break
            t += step
            frames_written += 1
    finally:
        if writer is not None:
            writer.release()
            print(f"Wrote {frames_written} frames → {args.out}")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
