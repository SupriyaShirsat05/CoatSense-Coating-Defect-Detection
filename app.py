"""
Command-line run of the full pipeline on one image.

    python app.py data/my_image.jpg
    python app.py data/my_image.jpg --model models/best_safenet_120ep.pt --conf 0.15
"""
import argparse
import json

from pipeline import DEFAULT_MODEL_PATH, run_pipeline

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("image")
    p.add_argument("--model", default=DEFAULT_MODEL_PATH)
    p.add_argument("--conf", type=float, default=0.15)
    a = p.parse_args()

    result = run_pipeline(a.image, a.model, a.conf)
    print(json.dumps(result, indent=2))
