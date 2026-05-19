# BEL_AUV_PROJECT

## 1. Project Overview
Explain the complete project clearly.

Example:
This project develops an AI-powered underwater object detection and classification system for Autonomous Underwater Vehicles (AUVs). The system detects underwater objects using YOLOv8 and classifies them using CNN/SVM models.

---

## 2. Objectives
- Detect underwater objects
- Improve underwater navigation
- Real-time object recognition
- AI-based marine analysis

---

## 3. Technologies Used

|  Technology  | Purpose              |
|--------------|----------------------|
| Python       | Core programming     |
| OpenCV       | Image processing     |
| YOLOv8       | Object detection     |
| PyTorch      | Deep learning        |
| scikit-learn | SVM classification   |
| NumPy        | Numerical operations |

---

## 4. System Architecture

Explain workflow:

Input Image
↓
Preprocessing
↓
YOLO Detection
↓
Feature Extraction
↓
CNN/SVM Classification
↓
Output Prediction

(Add architecture image later)

---

## 5. Dataset Information

Mention:
- Dataset source
- Number of images
- Annotation format
- YOLO labels used

Example:
- Total Images: 8000+
- Annotation Format: YOLO TXT
- Classes: Fish, Coral, Rock, etc.

---

## 6. Folder Structure

```text
BEL_AUV_PROJECT/
│
├── phase2_cnn/
├── dataset/
├── models/
├── runs/
├── README.md
├── requirements.txt
└── scripts
```

---

## 7. Installation

```bash
git clone https://github.com/pavan999-eng/BEL_AUV_PROJECT.git

cd BEL_AUV_PROJECT

pip install -r requirements.txt
```

---

## 8. Running the Project

### Train YOLO

```bash
python train.py
```

### Run Detection

```bash
python detect.py
```

### CNN Classification

```bash
python cnn_classifier.py
```

---

## 9. Results

Add:
- Accuracy
- Precision
- Recall
- Sample outputs
- Screenshots

Example:

| Metric | Value |
|--------|------|
| mAP50 | 92% |
| Precision | 89% |
| Recall | 91% |

---

## 10. Challenges Faced

- Underwater image noise
- Low visibility
- Dataset imbalance
- Real-time processing limitations

---

## 11. Future Improvements

- Real-time AUV integration
- Better underwater enhancement
- Transformer-based models
- Edge-device deployment

---

## 12. Author

Pavan Kumar Varanasi

GitHub:
https://github.com/pavan999-eng
