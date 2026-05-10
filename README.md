# Two-Wheeler Violation Detection and License Plate Recognition in Low-Light Conditions

> Using YOLOv11, EasyOCR, and Gemini API

[![Conference](https://img.shields.io/badge/Accepted-ICETIST%202026-blue?style=for-the-badge)]()
[![Framework](https://img.shields.io/badge/Framework-YOLOv11-orange?style=flat-square)]()
[![OCR](https://img.shields.io/badge/OCR-EasyOCR-green?style=flat-square)]()
[![AI](https://img.shields.io/badge/Fallback%20AI-Gemini%20API-purple?style=flat-square)]()

---

## 👤 Contact Details

**Mohammad Irfan**
Department of Information Technology
CBIT (Chaitanya Bharathi Institute of Technology), Hyderabad, India
📧 [mdirfan392007@gmail.com](mailto:mdirfan392007@gmail.com)

---

## 🏆 Achievements

- **Accepted at ICETIST 2026** — (Scopus-indexed) International Conference on Emerging Trends in Interdisciplinary Science and Technology

---

## 📄 Abstract

Detecting two-wheeler traffic violations in real-world scenarios is extremely challenging due to low light, motion blur, and varied camera angles. This project proposes a multi-stage deep learning framework using multiple YOLOv11 models to detect riders, helmets, two-wheelers, and license plates. A hybrid OCR pipeline with contrast adjustment, sharpening, and multi-angle recognition handles plate orientation variations. A Gemini API-based fallback mechanism retrieves text from images where standard OCR fails. The system demonstrates consistent performance under adverse conditions and is designed for use in smart traffic monitoring systems.

---

## 🎯 Objectives

- Use YOLOv11 models to accurately detect two-wheelers and riders under any environmental conditions including low light, motion blur, and varying angles.
- Detect two-wheeler violations — specifically **no helmet** and **triple riding** — using trained YOLOv11 models and a triple riding counting logic.
- Extract the correct license plate number from violated vehicles using a **hybrid OCR pipeline** that handles orientation, blur, and poor lighting.

---

## 🧪 Methodology

The system follows a multi-stage pipeline where each stage feeds into the next.

### Stage 1 — Two-Wheeler and Rider Detection

Two separate YOLOv11 models process the input image to detect motorcycles and their riders:

```
D_tw = YOLO_tw(I)   →   Two-Wheeler detections
D_r  = YOLO_r(I)    →   Rider detections
```

### Stage 2 — Violation Detection

**No Helmet Violation** — A binary check on the helmet detection output:
```
V_helmet = 1   if H = 0  (no helmet detected)
           0   otherwise
```

**Triple Riding Violation** — Counts detected riders per two-wheeler:
```
V_triple = 1   if Nr > 2  (more than 2 riders)
           0   otherwise
```

**Combined Violation Flag:**
```
V = V_helmet OR V_triple
```

### Stage 3 — License Plate Detection

A dedicated YOLOv11 model detects the license plate region on violated vehicles:
```
D_lp = YOLO_lp(I)
```

### Stage 4 — Image Enhancement

The detected plate crop is preprocessed before OCR:
- Scaling to a consistent resolution
- Contrast enhancement using **CLAHE**
- Sharpening to improve text edges

### Stage 5 — Multi-Angle OCR

OCR is applied at four orientations to handle tilted or rotated plates:
```
θ ∈ {0°, 90°, 180°, 270°}

T_θ = OCR(I_θ)
T*  = argmax_θ (Confidence(T_θ))
```
The result with the highest confidence score is selected as the final output.

### Stage 6 — Gemini API Fallback

When OCR output is missing or below acceptable confidence, the **Gemini Vision API** is used as a fallback to extract text from difficult or degraded plate images.

### Stage 7 — Text Post-Processing

Extracted text is cleaned and validated against the standard Indian license plate format:
```
[State Code] [District Code] [Series] [Number]
Example: TS 09 AB 1234
```

---

## 📊 Dataset

Custom datasets were collected covering varied lighting conditions, orientations, and traffic density:

| Task                    | Total Images | Training | Validation |
|-------------------------|:------------:|:--------:|:----------:|
| License Plate Detection | 1200         | 960      | 240        |
| Helmet Detection        | 500          | 400      | 100        |
| Two-Wheeler Detection   | 300          | 240      | 60         |
| Rider Detection         | 300          | 240      | 60         |

---

## 📈 Results

### Helmet Detection Model

| Metric                         | Score |
|--------------------------------|:-----:|
| mAP@0.5                        | 0.67  |
| Best F1-Score                  | 0.70  |
| Confidence Threshold (Best F1) | 0.49  |

Precision approaches 1.0 at high confidence thresholds (very few false positives), while recall decreases gradually — demonstrating a clean precision-recall trade-off that can be tuned per deployment requirement.

### License Plate Detection — OCR Method Comparison

| Method              | mAP@0.5 | Precision | Recall | F1-Score |
|---------------------|:-------:|:---------:|:------:|:--------:|
| EasyOCR             | 0.737   | 0.70      | 0.72   | 0.71     |
| CRNN-Based OCR      | 0.795   | 0.689     | 0.85   | 0.76     |
| **Proposed Method** | **0.91**| **0.90**  | **0.86**| **0.88** |

The proposed hybrid pipeline achieves the best scores across all metrics. The improvement comes from preprocessing (CLAHE + sharpening), multi-angle OCR selection, and the Gemini API fallback — making the system robust against low-brightness, tilted, and blurry plate images.

---

## 🛠️ Tech Stack

| Component               | Technology            |
|-------------------------|-----------------------|
| Object Detection        | YOLOv11 (Ultralytics) |
| OCR Engine              | EasyOCR               |
| AI Fallback             | Google Gemini API     |
| Image Preprocessing     | OpenCV (CLAHE)        |
| Deep Learning Framework | PyTorch               |
| Language                | Python                |

---

# Running the Project

## Step 1 — Clone the Repository

```bash
git clone https://github.com/MohammadIrfan01729/Two-wheeler_Violation_Detection_License_Plate_Recognition.git
```

---

## Step 2 — Upload the Project Folder to Google Drive

After cloning the repository, upload the complete project folder into your Google Drive without changing the folder structure.

The project is fully designed around:
- Google Colab
- Google Drive-mounted datasets
- Drive-based file paths

Maintaining the same folder structure is important for successful execution.

---

## Step 3 — Open Google Colab

Open Google Colab and load the required notebook files from Google Drive.

Main notebooks:
- `Training.ipynb`
- `Training2.ipynb`
- `Stage_wise_code.ipynb`

---

## Step 4 — Mount Google Drive

Run the following cell inside the notebook to mount Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

After authorization, Google Drive will be mounted successfully.

---

## Step 5 — Install Required Dependencies

All required package installation commands are already included inside the notebook files.

Run the installation cells sequentially.

Example:

```python
!pip install ultralytics
!pip install easyocr
!pip install google-generativeai
```

---

## Step 6 — Verify Dataset and Folder Paths

The project uses Google Drive paths directly inside the notebooks.

Ensure that:
- The uploaded folder structure remains unchanged
- Dataset paths inside the notebooks match the uploaded Drive location

---

## Step 7 — Run Training Notebooks

To train the YOLOv11 models:

### Run:
- `Training.ipynb`
- `Training2.ipynb`

These notebooks include:
- Dataset preparation
- Model training
- Metrics generation
- Validation
- Weights saving

---

## Step 8 — Run Complete Detection Pipeline

Open and run:

```text
Stage_wise_code.ipynb
```

This notebook performs:
- Two-wheeler detection
- Rider detection
- Helmet / no-helmet detection
- Triple riding detection
- License plate detection
- OCR recognition
- Gemini API fallback OCR
- Final output generation

---

## Step 9 — View Outputs

The generated outputs are stored inside the `output/` folder.

The outputs include:
- Stage-wise detection results
- Violation detection outputs
- License plate outputs
- OCR recognition results
- Final annotated images

---

# Important Notes

- Do not modify the folder structure unless corresponding paths inside the notebooks are updated.
- The project is optimized for Google Colab execution.
- All trained model weights are already included in the repository.
- The project works best with GPU enabled in Google Colab.

---

# Recommended Runtime

For better performance in Google Colab:

```text
Runtime → Change Runtime Type → GPU
```

Recommended:
- T4 GPU
- High RAM (optional)

---

# Supported Violations

The current system detects:
- No Helmet Violation
- Triple Riding Violation

---

# OCR Pipeline

The OCR pipeline includes:
- License plate cropping
- Image enhancement
- Multi-angle OCR
- Confidence-based text selection
- Gemini API fallback recognition

---

# Final Output

The final output contains:
- Annotated traffic violation image
- Detected violations
- Recognized license plate text
- Stage-wise processing outputs

---

## 🔭 Future Scope

- **Improved Helmet Detection** — Larger and more diverse training data to handle occlusion and varied rider poses.
- **Edge Deployment** — Optimise the pipeline for low-resource devices to enable real-time roadside enforcement.
- **Extended Violation Coverage** — Add detection for signal jumping, wrong-lane driving, and mobile phone usage while riding.
- **Enhanced OCR** — Fine-tune recognition models specifically for Indian regional license plate formats.
- **ITS Integration** — Connect with Intelligent Transportation Systems for centralised automated reporting and real-time enforcement.
- **Broader Generalisation** — Train and validate across different regions and environmental conditions for wider deployment.

---

<p align="center">
  Accepted at <strong>ICETIST 2026</strong> &nbsp;|&nbsp; Department of Information Technology, CBIT, Hyderabad
</p>
