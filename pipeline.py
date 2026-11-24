"""
CoatSense pipeline: detections -> one graded finding per defect type found.

analyze_detections() takes plain detections, so it can be tested without YOLO.
"""
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
for sub in ("detection", "severity", "grading", "audit", "assessment"):
    sys.path.append(os.path.join(ROOT, sub))

from grading_engine import GRADE_TABLES, defect_for_class, grade_defect  # noqa: E402
from service_life import assess_reuse  # noqa: E402
from severity_analyzer import (DEFAULT_MIN_SATURATION, boxes_mask, load_image,  # noqa: E402
                               measure, rust_pixel_mask)

DEFAULT_MODEL_PATH = os.path.join(ROOT, "models", "best_safenet_120ep.pt")

# BGR colours for drawing boxes
CLASS_COLOURS = {
    "corrosion": (30, 90, 220),
    "crack": (220, 120, 30),
    "flaking": (30, 180, 220),
    "blistering": (200, 60, 200),
}


def _iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def dedupe(dets: list, iou_threshold: float = 0.6) -> list:
    """Drop a box if it overlaps a higher-confidence box of the same class by more than the threshold."""
    kept = []
    for d in sorted(dets, key=lambda x: x["confidence"], reverse=True):
        if all(_iou(d["bbox"], k["bbox"]) <= iou_threshold for k in kept):
            kept.append(d)
    return kept


REVIEW_CONF = 0.30  # findings whose best detection is below this are flagged for human review


def analyze_detections(image_path: str, detections: list,
                       min_saturation: int = DEFAULT_MIN_SATURATION) -> dict:
    img = load_image(image_path)

    by_class = {}
    for d in detections:
        by_class.setdefault(d["class"], []).append(d)

    findings = []
    for class_name, dets in by_class.items():
        defect = defect_for_class(class_name)
        if defect is None:
            continue  # class has no grading table - skip rather than mis-grade
        dets = dedupe(dets)
        boxes = [d["bbox"] for d in dets]
        metric = GRADE_TABLES[defect]["metric"]

        m = measure(img, defect, metric, boxes, min_saturation)
        g = grade_defect(defect, m["value"])
        confs = [d["confidence"] for d in dets]

        findings.append({
            "class": class_name,
            "defect_type": defect,
            "n_detections": len(dets),
            "confidence_max": round(max(confs), 3),
            "confidence_mean": round(sum(confs) / len(confs), 3),
            "metric": metric,
            "measured_value": m["value"],
            "method": m["method"],
            "is_proxy": m["is_proxy"],
            "needs_review": max(confs) < REVIEW_CONF,
            "standard": g["standard"],
            "grade": g["grade"],
            "label": g["label"],
            "recommendation": g["recommendation"],
            "severity_index": g["severity_index"],
            "reuse_class": g["reuse_class"],
            "boxes": boxes,
        })

    if not findings:
        return {"image": image_path, "findings": [], "overall_grade": "No defects detected",
                "overall_condition": "No defects detected at this confidence threshold",
                "needs_review": False, "reuse": assess_reuse([])}

    worst = max(findings, key=lambda f: (f["severity_index"], f["confidence_max"]))
    return {
        "image": image_path,
        "findings": findings,
        "overall_grade": f"{worst['grade']} ({worst['class']})",
        # Use the grade table's own recommendation so the headline never disagrees with the detail below it
        "overall_condition": worst["recommendation"],
        "needs_review": any(f["needs_review"] for f in findings),
        "reuse": assess_reuse(findings, any(f["needs_review"] for f in findings)),
    }


def draw_findings(image_path: str, findings: list):
    """Return an RGB image with each finding's boxes drawn on it."""
    img = load_image(image_path)
    for f in findings:
        colour = CLASS_COLOURS.get(f["class"].lower(), (0, 255, 0))
        for x1, y1, x2, y2 in f["boxes"]:
            cv2.rectangle(img, (x1, y1), (x2, y2), colour, max(2, img.shape[1] // 300))
        if f["boxes"]:
            x1, y1 = f["boxes"][0][0], f["boxes"][0][1]
            cv2.putText(img, f"{f['class']} {f['grade']}", (x1 + 4, max(20, y1 + 24)),
                        cv2.FONT_HERSHEY_SIMPLEX, max(0.6, img.shape[1] / 1200), colour, 2)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def rust_overlay(image_path: str, findings: list, min_saturation: int = DEFAULT_MIN_SATURATION):
    """RGB image with the pixels counted as 'rust' highlighted, so the colour threshold can be checked by eye."""
    corr = [f for f in findings if f["defect_type"] == "rusting"]
    if not corr:
        return None
    img = load_image(image_path)
    boxes = [b for f in corr for b in f["boxes"]]
    counted = rust_pixel_mask(img, min_saturation) & boxes_mask(img.shape, boxes)
    out = img.copy()
    out[counted] = (0.4 * img[counted] + 0.6 * np.array([255, 255, 0])).astype(np.uint8)  # cyan highlight
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 255), 1)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def load_detector(model_path: str = DEFAULT_MODEL_PATH):
    from detector import Detector  # imported lazily so analysis can be tested without torch
    return Detector(model_path)


def run_pipeline(image_path: str, model_path: str = DEFAULT_MODEL_PATH, conf_threshold: float = 0.15,
                 min_saturation: int = DEFAULT_MIN_SATURATION) -> dict:
    detector = load_detector(model_path)
    return analyze_detections(image_path, detector.detect(image_path, conf_threshold), min_saturation)
