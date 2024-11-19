from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.yaml") # Build YOLOv11 nano from scratch

# Train the model
results = model.train(
    data="config.yaml",  # Path to dataset YAML
    epochs=1,  # Number of training epochs
    imgsz=640,  # Training image size
    device="cpu", # Single NVIDIA GPU (device=0), multiple NVIDIA GPUs (device=0,1), CPU (device=cpu)
)