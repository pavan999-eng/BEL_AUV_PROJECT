# BEL_AUV_PROJECT

## 1. Project Overview

This project develops an AI-powered underwater object detection and classification system for Autonomous Underwater Vehicles (AUVs). The system uses YOLOv8 for underwater object detection and CNN/SVM-based models for object classification and analysis.

The main goal of the project is to improve underwater navigation, marine monitoring, and real-time object recognition using deep learning and computer vision techniques.

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

text Input Image       ↓ Preprocessing       ↓ YOLOv8 Detection       ↓ Feature Extraction       ↓ CNN/SVM Classification       ↓ Output Prediction 

---

## 5. Dataset Information

- Dataset Source: Custom underwater dataset
- Total Images: 8000+
- Annotation Format: YOLO TXT
- Classes: Fish, Coral, Rock, Underwater Objects

---

## 6. Folder Structure

text BEL_AUV_PROJECT/ │ ├── phase2_cnn/ ├── dataset/ ├── models/ ├── runs/ ├── README.md ├── requirements.txt └── scripts/ 

---

## 7. Installation

Clone the repository:

bash git clone https://github.com/pavan999-eng/BEL_AUV_PROJECT.git 

Navigate to the project directory:

bash cd BEL_AUV_PROJECT 

Install dependencies:

bash pip install -r requirements.txt 

---

## 8. Running the Project

### Train YOLO Model

bash python train.py 

### Run Object Detection

bash python detect.py 

### CNN Classification

bash python cnn_classifier.py 

---

## 9. Results

| Metric | Value |
|--------|------|
| mAP50 | 92% |
| Precision | 89% |
| Recall | 91% |

### Sample Outputs

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

GitHub: https://github.com/pavan999-eng