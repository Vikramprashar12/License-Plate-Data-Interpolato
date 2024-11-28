import tkinter as tk
import tkinter.ttk as ttk
from tkinter import filedialog, messagebox
import os
import threading
import cv2
from ultralytics import YOLO
from sort.sort import *
from util import get_car, read_license_plate, write_csv
import numpy as np
import pandas as pd

class LicensePlateDetectorUI:
    def __init__(self, root):
        self.root = root
        self.root.title("License Plate Detector")
        self.root.geometry("600x400")

        # Configure root grid
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        # Create main frame with padding
        self.main_frame = ttk.Frame(root, padding="20")
        self.main_frame.grid(row=0, column=0, sticky="nsew")

        # Configure main frame grid
        self.main_frame.grid_columnconfigure(0, weight=1)

        # Title Label with custom style
        title_label = ttk.Label(
            self.main_frame,
            text="License Plate Detector",
            font=('Helvetica', 16, 'bold')
        )
        title_label.grid(row=0, column=0, pady=(0, 20))

        # File selection frame
        file_frame = ttk.Frame(self.main_frame)
        file_frame.grid(row=1, column=0, pady=(0, 15))

        # File selection button
        self.select_btn = ttk.Button(
            file_frame,
            text="Select MP4 File",
            command=self.select_file,
            width=20
        )
        self.select_btn.grid(row=0, column=0, padx=5)

        # Video info frame with border
        self.info_frame = ttk.LabelFrame(
            self.main_frame,
            text="Video Information",
            padding="10"
        )
        self.info_frame.grid(row=2, column=0, pady=(0, 15), sticky="ew")

        # Configure info frame grid
        self.info_frame.grid_columnconfigure(0, weight=1)

        # Video details labels with consistent spacing
        self.file_label = ttk.Label(self.info_frame, text="File: None")
        self.file_label.grid(row=0, column=0, sticky="w", pady=2)

        self.length_label = ttk.Label(self.info_frame, text="Length: 0:00")
        self.length_label.grid(row=1, column=0, sticky="w", pady=2)

        self.fps_label = ttk.Label(self.info_frame, text="FPS: 0")
        self.fps_label.grid(row=2, column=0, sticky="w", pady=2)

        # Process button with custom style
        self.process_btn = ttk.Button(
            self.main_frame,
            text="Process Video",
            command=self.start_processing,
            state="disabled",
            width=20
        )
        self.process_btn.grid(row=3, column=0, pady=(0, 15))

        # Progress frame with border
        self.progress_frame = ttk.LabelFrame(
            self.main_frame,
            text="Progress",
            padding="10"
        )
        self.progress_frame.grid(row=4, column=0, sticky="ew")

        # Configure progress frame grid
        self.progress_frame.grid_columnconfigure(0, weight=1)

        # Progress bar with increased width
        self.progress = ttk.Progressbar(
            self.progress_frame,
            mode='determinate',
            length=400
        )
        self.progress.grid(row=0, column=0, pady=(0, 10), sticky="ew")

        # Status and confidence labels with consistent styling
        self.status_label = ttk.Label(
            self.progress_frame,
            text="",
            font=('Helvetica', 10)
        )
        self.status_label.grid(row=1, column=0, pady=(0, 5))

        self.confidence_label = ttk.Label(
            self.progress_frame,
            text="",
            font=('Helvetica', 10)
        )
        self.confidence_label.grid(row=2, column=0)

        # Initialize variables
        self.video_path = None
        self.processing = False
        self.total_frames = 0
        self.fps = 0

        # Apply custom styling
        style = ttk.Style()
        style.configure('TButton', padding=5)
        style.configure('TLabelframe', padding=10)

        # Center the window on screen
        self.center_window()

    def center_window(self):
        """Center the window on the screen"""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')

    def select_file(self):
        self.video_path = filedialog.askopenfilename(filetypes=[("MP4 files", "*.mp4")])
        if self.video_path:
            # Get video properties
            cap = cv2.VideoCapture(self.video_path)
            self.fps = int(cap.get(cv2.CAP_PROP_FPS))
            self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            length_seconds = self.total_frames / self.fps
            minutes = int(length_seconds // 60)
            seconds = int(length_seconds % 60)

            # Update UI with video information
            self.file_label.config(text=f"File: {os.path.basename(self.video_path)}")
            self.length_label.config(text=f"Length: {minutes}:{seconds:02d}")
            self.fps_label.config(text=f"FPS: {self.fps}")
            self.process_btn.config(state="normal")
            cap.release()

    def start_processing(self):
        if not self.video_path or self.processing:
            return

        self.processing = True
        self.process_btn.config(state="disabled")
        self.select_btn.config(state="disabled")
        self.status_label.config(text="Processing video...")

        # Start processing in a separate thread
        thread = threading.Thread(target=self.process_video)
        thread.start()

    def process_video(self):
        try:
            results = {}
            mot_tracker = Sort()

            # Load models
            coco_model = YOLO('yolov8n.pt')
            license_plate_detector = YOLO('./models/license_plate_detector.pt')

            cap = cv2.VideoCapture(self.video_path)

            # Setup output video
            output_path = 'output_video.mp4'
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, self.fps,
                                  (int(cap.get(3)), int(cap.get(4))))

            frame_nmr = -1
            vehicles = [2, 3, 5, 7]
            ret = True

            while ret:
                frame_nmr += 1
                ret, frame = cap.read()
                if ret:
                    results[frame_nmr] = {}

                    try:
                        # Detect vehicles
                        detections = coco_model(frame, verbose=False)[0]
                        detections_ = []

                        for detection in detections.boxes.data.tolist():
                            x1, y1, x2, y2, score, class_id = detection
                            if int(class_id) in vehicles:
                                detections_.append([x1, y1, x2, y2, score])
                                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

                        # Track vehicles only if there are detections
                        if len(detections_) > 0:
                            track_ids = mot_tracker.update(np.asarray(detections_))
                        else:
                            track_ids = np.empty((0, 5))

                        try:
                            # Detect license plates
                            license_plates = license_plate_detector(frame, verbose=False)[0]
                            current_confidence = 0

                            for license_plate in license_plates.boxes.data.tolist():
                                x1, y1, x2, y2, score, class_id = license_plate

                                # Assign license plate to car
                                xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

                                if car_id != -1:
                                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

                                    # Ensure crop coordinates are within frame bounds
                                    y1, y2 = max(0, int(y1)), min(frame.shape[0], int(y2))
                                    x1, x2 = max(0, int(x1)), min(frame.shape[1], int(x2))

                                    license_plate_crop = frame[y1:y2, x1:x2, :]

                                    if license_plate_crop.size != 0:
                                        license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                                        _, license_plate_crop_thresh = cv2.threshold(
                                            license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

                                        license_plate_text, license_plate_text_score = read_license_plate(
                                            license_plate_crop_thresh)

                                        if license_plate_text is not None:
                                            current_confidence = max(current_confidence, license_plate_text_score)

                                            if car_id not in results[frame_nmr].keys():
                                                results[frame_nmr][car_id] = [license_plate_text]
                                            else:
                                                if license_plate_text not in results[frame_nmr][car_id]:
                                                    results[frame_nmr][car_id].append(license_plate_text)

                                            cv2.putText(
                                                frame,
                                                license_plate_text,
                                                (int(x1), int(y1) - 10),
                                                cv2.FONT_HERSHEY_SIMPLEX,
                                                1.2,
                                                (0, 0, 255),
                                                3
                                            )

                        except Exception as license_plate_exception:
                            print(f'License plate detection error: {license_plate_exception}')

                    except Exception as detection_exception:
                        print(f'Detection error: {detection_exception}')

                    # Write frame to output video
                    out.write(frame)

                    # Update progress bar
                    progress = ((frame_nmr + 1) / self.total_frames) * 100
                    self.progress['value'] = progress
                    self.root.update_idletasks()

            cap.release()
            out.release()

            # Save results to CSV
            write_csv(results, './output.csv')

            # Automatically play the output video after processing
            self.play_video(output_path)

            # Display information from output.csv
            self.display_output_csv_info()

            self.status_label.config(text="Video processing completed.")
        except Exception as processing_exception:
            self.status_label.config(text=f"Processing error: {processing_exception}")
        finally:
            self.processing = False
            self.process_btn.config(state="normal")
            self.select_btn.config(state="normal")

    def play_video(self, video_path):
        """Play the video using the default video player."""
        try:
            if os.name == 'nt':
                os.startfile(video_path)
            elif os.name == 'posix':
                os.system(f'open "{video_path}"')
            else:
                messagebox.showinfo("Info", f"Please play the video: {video_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not play the video: {e}")

    def display_output_csv_info(self):
        """Display information from output.csv in a message box."""
        try:
            df = pd.read_csv('./output.csv')
            info_text = df.to_string(index=False)
            messagebox.showinfo("Output CSV Information", info_text)
        except Exception as e:
            messagebox.showerror("Error", f"Could not read output.csv: {e}")

# Create the main window
root = tk.Tk()
app = LicensePlateDetectorUI(root)
root.mainloop()