import tkinter as tk
from tkinter import filedialog, messagebox
import tkinter.ttk as ttk
import subprocess
import cv2
from PIL import Image, ImageTk
import os
import threading

# Declare the global variable to store the thread reference
processing_thread = None

# Function to select video file and display its information


def select_video():
    input_file = filedialog.askopenfilename(title="Select Input Video", filetypes=[
                                            ("Video Files", "*.mp4 *.avi *.mkv")])

    if input_file:
        # Update video information labels
        video_info['text'] = f"File: {os.path.basename(input_file)}"
        video_length, fps = get_video_info(input_file)
        video_length_label['text'] = f"Length: {video_length}"
        video_fps_label['text'] = f"FPS: {fps}"

        # Save input video path for processing
        with open("video_path.txt", "w") as f:
            f.write(input_file)

        # Enable process button
        process_button.config(state=tk.NORMAL)

# Function to get video length and FPS


def get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return "N/A", "N/A"

    # Calculate video length
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    length = frames / fps
    cap.release()

    # Format the length into minutes and seconds
    minutes = int(length // 60)
    seconds = int(length % 60)
    length_str = f"{minutes:02}:{seconds:02}"

    return length_str, fps

# Function to run the processing pipeline


def run_pipeline():
    global processing_thread  # Use the global variable
    if processing_thread:
        return

    process_button.config(state=tk.DISABLED)
    select_button.config(state=tk.DISABLED)
    status_label.config(text="Processing video...")
    progress['value'] = 0
    progress.update_idletasks()

    # Start the processing thread
    processing_thread = threading.Thread(target=run_pipeline_thread)
    processing_thread.start()

# Function to run the pipeline in a separate thread


def run_pipeline_thread():
    try:
        update_progress(10, "Running main.py...")
        subprocess.run(["python", "main.py"], check=True)

        update_progress(40, "Running add_missing_data.py...")
        subprocess.run(["python", "interpolate.py"], check=True)

        update_progress(70, "Running visualize.py...")
        subprocess.run(["python", "visualize.py"], check=True)

        update_progress(100, "Video processing completed.")
        play_output_video()

    except subprocess.CalledProcessError as e:
        messagebox.showerror(
            "Error", f"An error occurred while running the pipeline:\n{str(e)}")
        update_progress(0, "An error occurred.")
    finally:
        global processing_thread
        processing_thread = None  # Reset thread reference to allow another run
        process_button.config(state=tk.NORMAL)
        select_button.config(state=tk.NORMAL)

# Function to update progress bar and status label


def update_progress(value, status_text):
    progress['value'] = value
    status_label.config(text=status_text)
    root.update_idletasks()

# Function to play the output video in the tkinter window


def play_output_video():
    output_file = "out.mp4"
    if not os.path.exists(output_file):
        messagebox.showerror(
            "File Not Found", f"{output_file} was not generated.")
        return

    cap = cv2.VideoCapture(output_file)
    if not cap.isOpened():
        messagebox.showerror("Error", "Could not open the output video.")
        return

    replay_button.pack_forget()  # Hide the replay button if visible

    # Set canvas and window size based on video dimensions while maintaining a fixed width of 700 pixels
    canvas_width = 700
    canvas_height = int(canvas_width * 9 / 16)  # Maintain 16:9 aspect ratio
    window_height_above_video = 400  # Estimated height of UI elements above video
    total_window_height = canvas_height + window_height_above_video

    # Set window size to fit the video and UI elements
    root.geometry(f"{canvas_width}x{total_window_height}")

    def update_frame():
        ret, frame = cap.read()
        if ret:
            # Resize the frame to fit the canvas dimensions while maintaining 16:9 aspect ratio
            frame_resized = cv2.resize(
                frame, (canvas_width, canvas_height), interpolation=cv2.INTER_AREA)

            # Convert frame to RGB for displaying in tkinter
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)

            # Update the canvas image
            canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
            canvas.image = imgtk

            # Schedule next frame
            root.after(10, update_frame)
        else:
            cap.release()
            # Show the replay button when video ends
            replay_button.pack(pady=10)

    # Set the canvas dimensions to match the fixed width and maintain 16:9 aspect ratio
    canvas.config(width=canvas_width, height=canvas_height)

    # Start updating frames
    update_frame()

# Function to replay the output video


def replay_video():
    play_output_video()


# Create the UI
root = tk.Tk()
root.title("License Plate Detector")

# Title Label
title_label = tk.Label(root, text="License Plate Detector",
                       font=("Helvetica", 16, "bold"), pady=10)
title_label.pack()

# Select MP4 File Button
select_button = tk.Button(root, text="Select MP4 File",
                          command=select_video, padx=20, pady=5)
select_button.pack(pady=10)

# Video Information Frame
video_info_frame = tk.Frame(root, relief=tk.SOLID,
                            borderwidth=1, padx=10, pady=10)
video_info_frame.pack(pady=10)

video_info_label = tk.Label(
    video_info_frame, text="Video Information", font=("Helvetica", 12, "bold"))
video_info_label.pack(anchor=tk.W)

video_info = tk.Label(video_info_frame, text="File: None", padx=5, pady=2)
video_info.pack(anchor=tk.W)

video_length_label = tk.Label(
    video_info_frame, text="Length: N/A", padx=5, pady=2)
video_length_label.pack(anchor=tk.W)

video_fps_label = tk.Label(video_info_frame, text="FPS: N/A", padx=5, pady=2)
video_fps_label.pack(anchor=tk.W)

# Process Video Button
process_button = tk.Button(root, text="Process Video",
                           command=run_pipeline, padx=20, pady=5, state=tk.DISABLED)
process_button.pack(pady=10)

# Progress Bar
progress = ttk.Progressbar(root, mode='determinate', length=400)
progress.pack(pady=10)

status_label = tk.Label(root, text="", font=("Helvetica", 10))
status_label.pack()

# Create a canvas to display the video
canvas_width = 700
canvas_height = int(canvas_width * 9 / 16)
# Maintain 16:9 aspect ratio
canvas = tk.Canvas(root, width=canvas_width, height=canvas_height)
canvas.pack()

# Replay Button
replay_button = tk.Button(root, text="Replay Video",
                          command=replay_video, padx=20, pady=5)
replay_button.pack(pady=10)
replay_button.pack_forget()  # Hide the replay button initially

# Start the Tkinter main loop
root.mainloop()
