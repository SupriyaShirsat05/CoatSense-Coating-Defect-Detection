"""
CoatSense severity analyzer (class-aware).

  - Corrosion  -> % of the IMAGE that is rust-coloured inside the detected boxes
  - Flaking    -> % of the image covered by detection boxes (upper-bound proxy)
  - Crack      -> number of distinct crack REGIONS (touching/overlapping boxes merged)
  - Blistering -> number of detected blisters

All area values are relative to the whole image, not to a single box.
"""
import cv2
import numpy as np

DEFAULT_MIN_SATURATION = 50  # raise it to ignore pale beige/cream surfaces


def load_image(image_path: str):
    # imdecode(fromfile) also works for Windows paths containing non-ASCII characters
    data = np.fromfile(image_path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return img


def boxes_mask(shape, boxes) -> np.ndarray:
    """Boolean mask that is True inside the UNION of all boxes (overlaps counted once)."""
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=bool)
    for x1, y1, x2, y2 in boxes:
        x1, x2 = max(0, int(x1)), min(w, int(x2))
        y1, y2 = max(0, int(y1)), min(h, int(y2))
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = True
    return mask


def rust_pixel_mask(img, min_saturation: int = DEFAULT_MIN_SATURATION) -> np.ndarray:
    """True where a pixel is rust-coloured (orange/brown/red-brown hues). OpenCV HSV: H 0-179."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for h_lo, h_hi in ((5, 25), (0, 5)):
        mask |= cv2.inRange(hsv, np.array([h_lo, min_saturation, 50]), np.array([h_hi, 255, 255]))
    return mask > 0


def rust_coverage(img, boxes, min_saturation: int = DEFAULT_MIN_SATURATION) -> dict:
    rust = rust_pixel_mask(img, min_saturation)
    inside = boxes_mask(img.shape, boxes)
    rust_px = int(np.count_nonzero(rust & inside))
    total = img.shape[0] * img.shape[1]
    return {
        "value": round(rust_px / total * 100, 2),
        "method": "HSV rust-colour mask inside detected boxes, as % of whole image",
        "is_proxy": False,
    }


def box_coverage(img, boxes) -> dict:
    inside = boxes_mask(img.shape, boxes)
    total = img.shape[0] * img.shape[1]
    return {
        "value": round(int(np.count_nonzero(inside)) / total * 100, 2),
        "method": "Detection-box coverage as % of whole image (upper bound - boxes over-estimate true area)",
        "is_proxy": True,
    }


def cluster_count(boxes, img_shape, gap_frac: float = 0.02) -> int:
    """
    Number of groups of touching/overlapping boxes. A detector often splits one long
    crack into many boxes; boxes closer than gap_frac * image size count as one region.
    """
    n = len(boxes)
    if n == 0:
        return 0
    gap = gap_frac * max(img_shape[:2])
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            a, b = boxes[i], boxes[j]
            if a[0] - gap < b[2] and b[0] - gap < a[2] and a[1] - gap < b[3] and b[1] - gap < a[3]:
                parent[find(i)] = find(j)
    return len({find(i) for i in range(n)})


def measure(img, defect_type: str, metric: str, boxes, min_saturation: int = DEFAULT_MIN_SATURATION) -> dict:
    """Pick the right measurement for this defect type."""
    if metric == "density":
        if defect_type == "cracking":
            return {"value": cluster_count(boxes, img.shape),
                    "method": "Number of distinct crack regions (touching/overlapping boxes merged; no physical scale available)",
                    "is_proxy": False}
        return {"value": len(boxes),
                "method": "Number of detected instances in the image (no physical scale available)",
                "is_proxy": False}
    if defect_type == "rusting":
        return rust_coverage(img, boxes, min_saturation)
    return box_coverage(img, boxes)
