"""
CoatSense detector: wraps the YOLO model and returns plain structured detections.
Answers "WHERE is the defect, and of which type?" - not "how severe is it?".
"""
from ultralytics import YOLO


class Detector:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)

    def detect(self, image_path: str, conf_threshold: float = 0.15) -> list:
        result = self.model(image_path, conf=conf_threshold, verbose=False)[0]
        detections = []
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append({
                "class": self.model.names[int(box.cls.item())],
                "confidence": round(box.conf.item(), 4),
                "bbox": [round(x1), round(y1), round(x2), round(y2)],
            })
        return detections
