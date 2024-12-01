import tkinter as tk
from tkinter import filedialog, messagebox
import subprocess
import cv2
from PIL import Image, ImageTk
import os

# Function to select video file and run the scripts
def run_pipeline():
    # Select video file
    input_file = filedialog.askopenfilename(title="Select Input Video", filetypes=[("Video Files", "*.mp4 *.avi *.mkv")])
    
    if not input_file:
        messagebox.showwarning("No File Selected", "Please select an input video file.")
        return

    # Save input video path to main.py
    with open("video_path.txt", "w") as f:
        f.write(input_file)

    try:
        # Run main.py
        subprocess.run(["python", "main.py"], check=True)

        # Run add_missing_data.py
        subprocess.run(["python", "interpolate.py"], check=True)

        # Run visualize.py
        subprocess.run(["python", "visualize.py"], check=True)

        # Play the output video in the tkinter window
        output_file = "out.mp4"
        if os.path.exists(output_file):
            play_video(output_file)
        else:
            messagebox.showerror("File Not Found", f"{output_file} was not generated.")

    except subprocess.CalledProcessError as e:
        messagebox.showerror("Error", f"An error occurred while running the pipeline:\n{str(e)}")

# Function to play the output video in the tkinter window
def play_video(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        messagebox.showerror("Error", "Could not open the output video.")
        return

    def update_frame():
        ret, frame = cap.read()
        if ret:
            # Convert frame to RGB for displaying in tkinter
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)

            # Update the canvas image
            canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
            canvas.image = imgtk

            # Schedule next frame
            root.after(10, update_frame)
        else:
            cap.release()

    # Create a canvas for the video
    canvas_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    canvas_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    canvas.config(width=canvas_width, height=canvas_height)

    # Start updating frames
    update_frame()

# Create the UI
root = tk.Tk()
root.title("License Plate Detection Pipeline")

# Set window size
root.geometry("800x600")

# Create a label
label = tk.Label(root, text="Select an input video file and run the detection pipeline.", padx=10, pady=10)
label.pack()

# Create a button to select input and run pipeline
run_button = tk.Button(root, text="Select Video and Run Pipeline", command=run_pipeline, padx=20, pady=10)
run_button.pack()

# Create a canvas to display the video
canvas = tk.Canvas(root, width=800, height=450)
canvas.pack()

# Start the Tkinter main loop
root.mainloop()