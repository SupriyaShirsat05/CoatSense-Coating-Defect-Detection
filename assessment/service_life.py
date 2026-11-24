"""
CoatSense reuse assessment and remaining-life estimate.

Two different things, deliberately kept apart:

1. assess_reuse()    - a verdict derived from the PHOTO (coating/surface grades). It says whether
                       the part looks reusable, not how much metal is left.
2. remaining_life()  - arithmetic on a THICKNESS READING supplied by the inspector. A photograph
                       cannot show metal loss, so years of life are never estimated from the image.
"""

CAVEAT = ("Based on photographs of the surface only. A photo cannot show loss of metal thickness, "
          "pitting depth or hidden damage. Final reuse decisions need thickness measurement and "
          "engineer sign-off. Thresholds are placeholders.")

REUSE_CLASSES = {
    "A": ("Reusable as is",
          "No re-coating needed now. Continue routine inspection."),
    "B": ("Reusable after re-coating",
          "Clean, prepare the surface and re-coat before further service."),
    "C": ("Conditionally reusable",
          "Measure remaining metal thickness first. Re-coat only if enough sound metal remains."),
    "D": ("Not recommended for reuse until assessed",
          "Do not return to service until an engineer has assessed section loss and structural soundness."),
}
ORDER = ["A", "B", "C", "D"]


def assess_reuse(findings: list, needs_review: bool = False) -> dict:
    if not findings:
        return {"class": "A", "status": "No visible defects detected",
                "action": "The photo shows no detectable coating defect. Confirm with a visual check.",
                "basis": [], "provisional": False, "override": False, "caveat": CAVEAT}
    worst = max((f["reuse_class"] for f in findings), key=ORDER.index)
    status, action = REUSE_CLASSES[worst]
    return {"class": worst, "status": status, "action": action,
            "basis": [f"{f['class']} {f['grade']} ({f['standard']})" for f in findings if f["reuse_class"] == worst],
            "provisional": bool(needs_review), "override": False, "caveat": CAVEAT}


def remaining_life(original_mm: float, measured_mm: float, minimum_mm: float, age_years: float) -> dict:
    """
    Standard corrosion-rate method used in industrial inspection practice:
        rate      = (original - measured) / years in service
        remaining = (measured - minimum required) / rate
    Assumes the average rate so far continues. Local pitting and a changing environment are not captured.
    """
    if min(original_mm, measured_mm, minimum_mm, age_years) <= 0:
        return {"valid": False, "error": "All four values must be greater than zero."}
    if measured_mm > original_mm:
        return {"valid": False, "error": "Measured thickness is greater than the original thickness; check the readings."}
    if minimum_mm >= original_mm:
        return {"valid": False, "error": "Minimum required thickness must be below the original thickness."}

    rate = (original_mm - measured_mm) / age_years
    out = {"valid": True, "corrosion_rate_mm_per_year": round(rate, 3),
           "original_mm": original_mm, "measured_mm": measured_mm, "minimum_mm": minimum_mm, "age_years": age_years}
    if measured_mm <= minimum_mm:
        out.update({"status": "AT_OR_BELOW_MINIMUM", "remaining_years": 0.0, "reinspect_years": None})
    elif rate == 0:
        out.update({"status": "NO_MEASURABLE_LOSS", "remaining_years": None, "reinspect_years": None})
    else:
        remaining = (measured_mm - minimum_mm) / rate
        # common inspection convention: re-inspect within half of the estimated remaining life
        out.update({"status": "OK", "remaining_years": round(remaining, 1), "reinspect_years": round(remaining / 2, 1)})
    return out


def apply_thickness_override(reuse: dict, life: dict) -> dict:
    """Metal at or below its minimum thickness overrides whatever the photo suggests."""
    if life and life.get("valid") and life.get("status") == "AT_OR_BELOW_MINIMUM":
        return {**reuse, "class": "D", "status": "Not fit for reuse: at or below minimum thickness",
                "action": "Measured thickness is at or below the minimum required. Repair or replace; do not return to service.",
                "basis": reuse["basis"] + ["Measured thickness at or below minimum"], "override": True}
    return reuse
