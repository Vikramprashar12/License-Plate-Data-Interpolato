import tkinter as tk
from tkinter import filedialog, messagebox
import subprocess
import cv2
from PIL import Image, ImageTk
import os

# Function to select video file and display its information
def select_video():
    input_file = filedialog.askopenfilename(title="Select Input Video", filetypes=[("Video Files", "*.mp4 *.avi *.mkv")])

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
    try:
        # Run main.py
        subprocess.run(["python", "main.py"], check=True)
        update_progress(30)

        # Run add_missing_data.py
        subprocess.run(["python", "interpolate.py"], check=True)
        update_progress(60)

        # Run visualize.py
        subprocess.run(["python", "visualize.py"], check=True)
        update_progress(100)

        # Play the output video in the tkinter window
        output_file = "out.mp4"
        if os.path.exists(output_file):
            play_video(output_file)
        else:
            messagebox.showerror("File Not Found", f"{output_file} was not generated.")

    except subprocess.CalledProcessError as e:
        messagebox.showerror("Error", f"An error occurred while running the pipeline:\n{str(e)}")

# Function to update progress bar
def update_progress(value):
    progress_var.set(value)
    root.update_idletasks()

# Function to play the output video in the tkinter window
def play_video(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        messagebox.showerror("Error", "Could not open the output video.")
        return

    replay_button.pack_forget()  # Hide the replay button if visible

    def update_frame():
        ret, frame = cap.read()
        if ret:
            # Get the canvas dimensions while maintaining 16:9 aspect ratio
            canvas_width = canvas.winfo_width()
            canvas_height = int(canvas_width * 9 / 16)

            # Resize the frame to fit the canvas dimensions
            frame_resized = cv2.resize(frame, (canvas_width, canvas_height), interpolation=cv2.INTER_AREA)

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
            replay_button.pack(pady=10)  # Show the replay button when video ends

    # Set the canvas dimensions to maintain a 16:9 aspect ratio
    canvas_width = root.winfo_width()
    canvas_height = int(canvas_width * 9 / 16)  # 16:9 aspect ratio
    canvas.config(width=canvas_width, height=canvas_height)

    # Start updating frames
    update_frame()

# Function to replay the output video
def replay_video():
    output_file = "out.mp4"
    if os.path.exists(output_file):
        play_video(output_file)

# Create the UI
root = tk.Tk()
root.title("License Plate Detector")

# Set window size
root.geometry("800x600")

# Title Label
title_label = tk.Label(root, text="License Plate Detector", font=("Helvetica", 16, "bold"), pady=10)
title_label.pack()

# Select MP4 File Button
select_button = tk.Button(root, text="Select MP4 File", command=select_video, padx=20, pady=5)
select_button.pack(pady=10)

# Video Information Frame
video_info_frame = tk.Frame(root, relief=tk.SOLID, borderwidth=1, padx=10, pady=10)
video_info_frame.pack(pady=10)

video_info_label = tk.Label(video_info_frame, text="Video Information", font=("Helvetica", 12, "bold"))
video_info_label.pack(anchor=tk.W)

video_info = tk.Label(video_info_frame, text="File: None", padx=5, pady=2)
video_info.pack(anchor=tk.W)

video_length_label = tk.Label(video_info_frame, text="Length: N/A", padx=5, pady=2)
video_length_label.pack(anchor=tk.W)

video_fps_label = tk.Label(video_info_frame, text="FPS: N/A", padx=5, pady=2)
video_fps_label.pack(anchor=tk.W)

# Process Video Button
process_button = tk.Button(root, text="Process Video", command=run_pipeline, padx=20, pady=5, state=tk.DISABLED)
process_button.pack(pady=10)

# Progress Bar
progress_var = tk.IntVar()
progress_bar = tk.Label(root, text="Progress", font=("Helvetica", 12), pady=5)
progress_bar.pack()
progress = tk.Scale(root, variable=progress_var, from_=0, to=100, orient="horizontal", length=400)
progress.pack(pady=10)

# Create a canvas to display the video
canvas = tk.Canvas(root, width=800, height=int(800 * 9 / 16))  # Maintain 16:9 aspect ratio
canvas.pack(fill=tk.BOTH, expand=True)

# Replay Button
replay_button = tk.Button(root, text="Replay Video", command=replay_video, padx=20, pady=5)
replay_button.pack(pady=10)
replay_button.pack_forget()  # Hide the replay button initially

# Start the Tkinter main loop
root.mainloop()