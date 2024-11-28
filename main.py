from ultralytics import YOLO
import cv2
import numpy as np
import util
from sort.sort import *
from util import get_car, read_license_plate, write_csv

results = {}

mot_tracker = Sort()

# Load models
coco_model = YOLO('yolov8n.pt')  # Model for general object detection
# Model for license plate detection
license_plate_detector = YOLO('./models/license_plate_detector.pt')

# Load video
cap = cv2.VideoCapture('./sample.mp4')

vehicles = [2, 3, 5, 7]  # Vehicle class IDs: car, motorcycle, bus, truck

# Create a slider window
cv2.namedWindow('License Plate Processing')

# Callback for slider (does nothing but is required for the trackbar)


def nothing(x):
    pass


# Add a slider to control threshold value
cv2.createTrackbar('Threshold', 'License Plate Processing', 110, 255, nothing)

# Pause control
pause = False

# Read frames
frame_nmr = -1
ret = True
while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if ret:
        results[frame_nmr] = {}

        # Check if paused
        if pause:
            while True:
                key = cv2.waitKey(1)
                if key == ord('p'):  # Resume if 'p' is pressed
                    pause = False
                    break
                elif key == ord('q'):  # Quit if 'q' is pressed
                    cap.release()
                    cv2.destroyAllWindows()
                    exit()

        # Detect vehicles
        detections = coco_model(frame)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection

            # Draw bounding box for every detected object (only vehicles visually shown)
            if int(class_id) in vehicles:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(
                    x2), int(y2)), (0, 255, 0), 2)  # Green for cars
                detections_.append([x1, y1, x2, y2, score])

        # Track vehicles (all vehicles kept)
        track_ids = mot_tracker.update(np.asarray(detections_))

        # Detect license plates
        license_plates = license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            # Draw bounding box for all license plates
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(
                y2)), (0, 0, 255), 2)  # Red for license plates

            license_plate_crop = frame[int(y1):int(y2), int(
                x1): int(x2), :]  # Crop license plate

            # Process license plate
            license_plate_crop_gray = cv2.cvtColor(
                license_plate_crop, cv2.COLOR_BGR2GRAY)

            # Get current threshold value from the slider
            current_threshold = cv2.getTrackbarPos(
                'Threshold', 'License Plate Processing')
            _, license_plate_crop_thresh = cv2.threshold(
                license_plate_crop_gray, 140, 255, cv2.THRESH_BINARY)

            # Ensure all images have the same size
            target_size = (200, 100)
            resized_crop = cv2.resize(
                license_plate_crop, target_size, interpolation=cv2.INTER_AREA)
            resized_gray = cv2.resize(
                license_plate_crop_gray, target_size, interpolation=cv2.INTER_AREA)
            resized_thresh = cv2.resize(
                license_plate_crop_thresh, target_size, interpolation=cv2.INTER_AREA)

            # Convert grayscale and binary images to BGR for consistent visualization
            gray_bgr = cv2.cvtColor(resized_gray, cv2.COLOR_GRAY2BGR)
            thresh_bgr = cv2.cvtColor(resized_thresh, cv2.COLOR_GRAY2BGR)

            # Combine all steps into one image
            combined_view = cv2.hconcat(
                [resized_crop, gray_bgr, thresh_bgr])
            cv2.imshow('License Plate Processing', combined_view)

            # Read license plate text
            license_plate_text, license_plate_text_score = read_license_plate(
                license_plate_crop_thresh)
            print(license_plate_text, license_plate_text_score)

            if license_plate_text is not None:
                results[frame_nmr][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                              'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                'text': license_plate_text,
                                                                'bbox_score': score,
                                                                'text_score': license_plate_text_score}}

        # Display the frame with all bounding boxes
        cv2.imshow('Frame with Bounding Boxes', frame)

        # Pause control
        key = cv2.waitKey(1)
        if key == ord('p'):  # Pause if 'p' is pressed
            pause = True
        elif key == ord('q'):  # Quit if 'q' is pressed
            break

# Release resources
cap.release()
cv2.destroyAllWindows()

# Write results to CSV
write_csv(results, './test.csv')
