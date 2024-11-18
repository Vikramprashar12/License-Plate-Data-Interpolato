from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.yaml") #Build YOLOv11 nano from scratch

# Train the model
train_results = model.train(
    data="coco8.yaml",  # path to dataset YAML
    epochs=100,  # number of training epochs
    imgsz=640,  # training image size
)