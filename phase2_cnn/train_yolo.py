from ultralytics import YOLO

# Load YOLOv8 Nano Model
model = YOLO("yolov8n.pt")

# Train Model
model.train(
    data="dataset.yaml",
    epochs=25,
    imgsz=640,
    batch=8,
    device="mps"
)