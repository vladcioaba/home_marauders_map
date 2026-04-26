"""SQLite writer for per-person position history.

Rate-limited to at most one row per `min_interval_s` per global ID so the DB
stays small at 30+ fps. Query the output with the `sqlite3` CLI or DB Browser.

Schema:
  events(t_s REAL, global_id INT, cam_id TEXT, x_m REAL, y_m REAL, room TEXT)
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from marauders_map.house import House
from marauders_map.tracker_global import GlobalMarker


class TrackLogger:
    def __init__(self, db_path: Path, min_interval_s: float = 1.0) -> None:
        self.min_interval_s = min_interval_s
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path))
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS events (
                t_s       REAL    NOT NULL,
                global_id INTEGER NOT NULL,
                cam_id    TEXT    NOT NULL,
                x_m       REAL    NOT NULL,
                y_m       REAL    NOT NULL,
                room      TEXT
            );
            CREATE INDEX IF NOT EXISTS events_gid_t ON events (global_id, t_s);
            CREATE INDEX IF NOT EXISTS events_t     ON events (t_s);
            """
        )
        self._conn.commit()
        self._last: dict[int, float] = {}

    def log(self, t: float, markers: list[GlobalMarker], house: House) -> None:
        rows = []
        for m in markers:
            if t - self._last.get(m.global_id, 0.0) < self.min_interval_s:
                continue
            self._last[m.global_id] = t
            room = house.room_at(m.floor_xy)
            rows.append((t, m.global_id, m.cam_id, m.floor_xy[0], m.floor_xy[1], room))
        if rows:
            self._conn.executemany(
                "INSERT INTO events (t_s, global_id, cam_id, x_m, y_m, room) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                rows,
            )
            self._conn.commit()

    def close(self) -> None:
        self._conn.close()
