# BEL_AUV_PROJECT

## 1. Project Overview

BEL_AUV_PROJECT is an AI-powered underwater object detection and classification system designed for Autonomous Underwater Vehicles (AUVs).

The project uses:

- YOLOv8 for underwater object detection
- CNN/SVM models for classification
- OpenCV for image processing
- Deep learning techniques for marine environment analysis

The goal is to improve:

- Underwater navigation
- Marine monitoring
- Real-time object recognition
- Autonomous underwater operations

---

## 2. Objectives

- Detect underwater objects accurately
- Improve underwater navigation systems
- Perform real-time object recognition
- Enable AI-based marine environment analysis
- Support Autonomous Underwater Vehicle (AUV) operations

---

## 3. Technologies Used

| Technology | Purpose |
|------------|---------|
| Python | Core programming |
| OpenCV | Image processing |
| YOLOv8 | Object detection |
| PyTorch | Deep learning |
| scikit-learn | SVM classification |
| NumPy | Numerical operations |

---

## 4. System Architecture

```text
Input Image
     ↓
Preprocessing
     ↓
YOLOv8 Detection
     ↓
Feature Extraction
     ↓
CNN/SVM Classification
     ↓
Output Prediction
```

---

## 5. Dataset Information

- Dataset Source: Custom underwater dataset
- Total Images: 8000+
- Annotation Format: YOLO TXT
- Classes:
  - Fish
  - Coral
  - Rock
  - Underwater Objects

---

## 6. Folder Structure

```text
BEL_AUV_PROJECT/
│
├── deployment/
├── phase1_ml/
├── phase2_cnn/
├── dataset/
├── models/
├── runs/
├── scripts/
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 7. Installation

### Clone the Repository

```bash
git clone https://github.com/pavan999-eng/BEL_AUV_PROJECT.git
```

### Navigate to Project Directory

```bash
cd BEL_AUV_PROJECT
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 8. Running the Project

### Train YOLO Model

```bash
python train.py
```

### Run Object Detection

```bash
python detect.py
```

### Run CNN Classification

```bash
python cnn_classifier.py
```

---

## 9. Results

| Metric | Value |
|--------|------|
| mAP50 | 92% |
| Precision | 89% |
| Recall | 91% |

### Outputs

- Underwater object detection results
- YOLO prediction outputs
- CNN classification outputs

---

## 10. Challenges Faced

- Underwater image noise
- Low visibility conditions
- Dataset imbalance
- Real-time processing limitations

---

## 11. Future Improvements

- Real-time AUV integration
- Advanced underwater image enhancement
- Transformer-based detection models
- Edge-device deployment optimization

---

## 12. Author

### Pavan Kumar Varanasi

GitHub:  
https://github.com/pavan999-eng

---

## 13. License

This project is developed for research and educational purposes.
