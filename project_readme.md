# CoatSense

**AI-Based Detection, Classification, and Provisional Grading of Protective Coating Defects**

CoatSense is a computer-implemented inspection support prototype designed for coated steel surfaces (such as naval and defence platforms). It combines object detection, defect-specific severity quantification, configurable standards-linked grading, human-in-the-loop confidence review, and timestamped audit logging in an interactive Streamlit dashboard.

---

## Key Features

- **Multi-Class Defect Detection:** Uses a fine-tuned `YOLOv8n` detector (`640 × 640` input, 120 epochs) to identify four primary coating defect classes:
  - **Corrosion**
  - **Cracking**
  - **Flaking**
  - **Blistering**
- **Defect-Specific Severity Quantification:** Avoids relying solely on raw bounding-box area by applying tailored post-processing per class:
  - **Corrosion:** HSV rust-pixel masking within detected corrosion regions (`%` of total image area, with adjustable minimum saturation $S_{\min}$ to reject pale/cream paints).
  - **Cracking:** Spatial union–find region merging using a gap proportional to image dimensions to group fragmented detections of the same crack before counting.
  - **Flaking:** Union of detected flaking bounding regions reported explicitly as an upper-bound coverage estimate (`%` of total image area).
  - **Blistering:** Unique instance counting after same-class duplicate suppression.
- **Provisional ISO 4628-Informed Rule Engine:** Maps defect measurements to severity grades, recommended maintenance actions, and reuse classes (`A`–`D`) using editable JSON rule tables linked to ISO 4628 Parts 2–5, with a normalized severity index ($s = i / (K - 1)$) to select the headline finding.
- **Human-in-the-Loop Confidence Review:** Highlights low-confidence findings ($c_{\max} < 0.30$) with review warnings and marks associated reuse decisions as tentative so final sign-off remains with the inspector.
- **Remaining-Life & Audit Logging:** Supports optional steel thickness inputs to estimate corrosion rate, remaining life, and suggested re-inspection interval, while storing every inspection as a timestamped record in SQLite.
- **Interactive Streamlit Dashboard:** Provides image upload, adjustable detection/colour thresholds, visual overlays (including counted rust-pixel highlights), and searchable inspection history.

---

## Project Structure

```text
CoatSense/
├── app.py                     # Streamlit dashboard entry point
├── models/                    # Place trained YOLO weights here (e.g., best_safenet_120ep.pt)
├── rules/                     # Editable JSON grading rule tables (ISO 4628-informed)
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

*(Note: Adjust file/folder names above if your local entry script or rule directory uses different filenames.)*

---

## Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/<your-username>/<your-repo-name>.git
   cd <your-repo-name>
   ```

2. **Create and activate a virtual environment (recommended):**
   ```bash
   python -m venv venv
   # On Windows (PowerShell):
   .\venv\Scripts\Activate.ps1
   # On Linux/macOS:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Place the trained model weights locally:**
   Because `.pt` weight files are excluded from Git version control, place your trained checkpoint file inside the `models/` directory before running the app:
   ```text
   models/best_safenet_120ep.pt
   ```

---

## Running the Application

Launch the Streamlit inspection dashboard from the project root:

```bash
streamlit run app.py
```

Once open in your browser:
1. Select `best_safenet_120ep.pt` in the model selector.
2. Adjust the detection confidence threshold and minimum HSV saturation ($S_{\min}$) as needed.
3. Upload a coated-surface image (JPEG/PNG) and run the inspection to view overlays, provisional grades, and saved audit logs.

---

## Model Evaluation Summary

- **Training Split (`2,996` images, `9,572` instances):** `78.43%` mAP@0.50 (`54.7%` mAP@0.50:0.95), `80.9%` Precision, `68.1%` Recall.
- **Unseen Validation Split (`1,311` images, `3,678` instances):** `43.11%` mAP@0.50 (`23.68%` mAP@0.50:0.95), `57.1%` Precision, `41.2%` Recall.

> **Disclaimer:** CoatSense is a research and engineering prototype intended for visual decision support and triage. Severity thresholds are provisional and are not a substitute for official ISO-certified inspection or physical scale calibration.