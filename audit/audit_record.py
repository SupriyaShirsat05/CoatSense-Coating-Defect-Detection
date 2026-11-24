"""
CoatSense audit records: every inspection is saved to SQLite with a unique ID and timestamp.
Stores per-class findings, the reuse assessment and (optionally) the thickness-based life estimate.
Older databases are upgraded automatically the first time they are opened.
"""
import json
import os
import sqlite3
from datetime import datetime, timedelta, timezone

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "inspections.db")


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS inspections (
            inspection_id   TEXT PRIMARY KEY,
            timestamp       TEXT NOT NULL,
            image_path      TEXT NOT NULL,
            asset_id        TEXT,
            location_tag    TEXT,
            detections_json TEXT NOT NULL,
            overall_grade   TEXT,
            reuse_json      TEXT,
            life_json       TEXT
        )
    """)
    existing = [r[1] for r in conn.execute("PRAGMA table_info(inspections)")]
    for col in ("reuse_json", "life_json"):          # upgrade databases created by earlier versions
        if col not in existing:
            conn.execute(f"ALTER TABLE inspections ADD COLUMN {col} TEXT")
    conn.commit()
    conn.close()


def _load(value):
    return json.loads(value) if value else None


def _row_to_record(r) -> dict:
    return {
        "inspection_id": r["inspection_id"], "timestamp": r["timestamp"], "image_path": r["image_path"],
        "asset_id": r["asset_id"], "location_tag": r["location_tag"],
        "detections": json.loads(r["detections_json"]),   # per-class findings
        "overall_grade": r["overall_grade"],
        "reuse": _load(r["reuse_json"]), "life": _load(r["life_json"]),
    }


def save_inspection(pipeline_result: dict, asset_id: str = "Unspecified",
                    location_tag: str = "Unspecified", life: dict = None) -> dict:
    init_db()
    inspection_id = f"INS-{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}"
    timestamp = datetime.now(timezone.utc).isoformat()
    findings = pipeline_result.get("findings", [])
    overall = pipeline_result.get("overall_grade", "No defects detected")
    reuse = pipeline_result.get("reuse")

    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO inspections (inspection_id, timestamp, image_path, asset_id, location_tag, "
        "detections_json, overall_grade, reuse_json, life_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (inspection_id, timestamp, pipeline_result.get("image", ""), asset_id, location_tag,
         json.dumps(findings), overall, json.dumps(reuse) if reuse else None, json.dumps(life) if life else None),
    )
    conn.commit()
    conn.close()
    return {"inspection_id": inspection_id, "timestamp": timestamp, "image_path": pipeline_result.get("image", ""),
            "asset_id": asset_id, "location_tag": location_tag, "detections": findings,
            "overall_grade": overall, "reuse": reuse, "life": life}


def clear_all_inspections() -> int:
    init_db()
    conn = sqlite3.connect(DB_PATH)
    count = conn.execute("SELECT COUNT(*) FROM inspections").fetchone()[0]
    conn.execute("DELETE FROM inspections")
    conn.commit()
    conn.close()
    return count


def get_inspections_by_date_range(days_back: int = None, start_date: str = None, end_date: str = None) -> list:
    init_db()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    if days_back is not None:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days_back)).isoformat()
        rows = conn.execute("SELECT * FROM inspections WHERE timestamp >= ? ORDER BY timestamp DESC", (cutoff,)).fetchall()
    elif start_date and end_date:
        rows = conn.execute("SELECT * FROM inspections WHERE date(timestamp) BETWEEN ? AND ? ORDER BY timestamp DESC",
                            (start_date, end_date)).fetchall()
    else:
        rows = conn.execute("SELECT * FROM inspections ORDER BY timestamp DESC").fetchall()
    conn.close()
    return [_row_to_record(r) for r in rows]


def get_all_inspections() -> list:
    return get_inspections_by_date_range()
