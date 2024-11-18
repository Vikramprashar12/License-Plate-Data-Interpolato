from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.yaml") #Build YOLOv11 nano from scratch

# Train the model
results = model.train(
    data="config.yaml",  # path to dataset YAML
    epochs=1,  # number of training epochs
    imgsz=640,  # training image size
    device=0,
)