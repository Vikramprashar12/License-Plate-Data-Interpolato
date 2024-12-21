# License Plate Data Interpolator

## Project Description
This project implements a License Plate Recognition System using **YOLO** object detection models. The system processes video frames to detect vehicles, track them, and recognize license plates. The primary goals are:
- **Vehicle Detection**: Identify vehicles such as cars, motorcycles, buses, and trucks.
- **License Plate Detection**: Detect license plates associated with identified vehicles.
- **Text Extraction**: Extract text from detected license plates.

## Problem Statement
Accurate license plate recognition is essential for traffic management, law enforcement, and automated parking systems. This project tackles the challenges of:
1. Detecting vehicles in real-time.
2. Accurately detecting and aligning license plates.
3. Extracting readable text from license plates under varying conditions.

## Data Sources
- **YOLO Pretrained Model (`yolov8n.pt`)**: Used for general object detection.
- **Custom License Plate Detection Model (`license_plate_detector.pt`)**: Trained for detecting license plates.

The input is a video file specified when running the ui.py program via the batch file.

## Code Explanation

### Workflow
1. **Load Models**:
   - Load the COCO model (`yolov8n.pt`) for general object detection.
   - Load a custom license plate detection model for identifying license plates.

2. **Read Video**:
   - The video is read frame by frame for processing.

3. **Vehicle Detection**:
   - Detect vehicles using YOLO.
   - Filter detections for specific vehicle classes: cars, motorcycles, buses, and trucks.

4. **Vehicle Tracking**:
   - Use the `Sort` algorithm for tracking vehicles across frames.

5. **License Plate Detection and Processing**:
   - Detect license plates within vehicle bounding boxes.
   - Align, gray-scale, and threshold license plate images for preprocessing.
   - Resize and display steps for visualization.

6. **License Plate Recognition**:
   - Extract text from license plates using OCR methods.
   - Associate recognized text with corresponding vehicles.

7. **Results Storage**:
   - Store vehicle and license plate details in a dictionary (`results`).
   - Save results to a CSV file (`test.csv`).

### Output
- Visualized bounding boxes for detected vehicles and license plates.
- Combined view of license plate processing steps.
- CSV file (`test.csv`) containing:
  - Vehicle bounding boxes.
  - License plate bounding boxes.
  - Recognized license plate text and confidence scores.

## How to Run

### Prerequisites
- Install required Python libraries:
  ```bash
  pip install -r requirements.txt
  ```
- Double click on the Run.bat file which will launch the ui.py functionality. 
