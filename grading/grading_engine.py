"""
CoatSense standards mapping engine.
Maps a measured value to a named grade. Ranges are half-open [min, max) and
contiguous, so no value can fall between two grades.
"""
import json
import os
from datetime import datetime, timezone

GRADES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "iso4628_grades.json")

with open(GRADES_PATH, "r", encoding="utf-8") as f:
    GRADE_TABLES = json.load(f)

# Model class name (lowercase) -> key in the grade tables
CLASS_TO_DEFECT = {
    "corrosion": "rusting",
    "crack": "cracking",
    "flaking": "peeling",
    "blistering": "blistering",
}


def defect_for_class(class_name: str):
    return CLASS_TO_DEFECT.get(class_name.lower().strip())


def grade_defect(defect_type: str, value: float) -> dict:
    defect_type = defect_type.lower().strip()
    if defect_type not in GRADE_TABLES:
        raise ValueError(f"Unknown defect type '{defect_type}'. Supported: {list(GRADE_TABLES)}")

    table = GRADE_TABLES[defect_type]
    rows = table["grades"]
    base = {
        "defect_type": defect_type,
        "standard": table["standard"],
        "metric": table["metric"],
        "unit": table["unit"],
        "measured_value": value,
        "graded_at": datetime.now(timezone.utc).isoformat(),
    }

    for i, row in enumerate(rows):
        if row["min"] <= value < row["max"]:
            base.update({
                "grade": row["grade"],
                "label": row["label"],
                "recommendation": row["recommendation"],
                "reuse_class": row["reuse"],
                # 0.0 = best grade, 1.0 = worst grade in this table; lets us
                # compare severity across different standards.
                "severity_index": round(i / (len(rows) - 1), 3),
            })
            return base

    base.update({
        "grade": "UNDETERMINED",
        "label": "Value outside the grading range - flagged for manual review",
        "recommendation": "Route to human inspector",
        "reuse_class": "D",
        "severity_index": 1.0,
    })
    return base
