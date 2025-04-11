#!/usr/bin/env python3
import os
import cv2
import numpy as np
import gradio as gr
import tempfile
import subprocess
from pathlib import Path
import time
import shutil
import people_counter
from people_counter import LineCounter
from ultralytics import YOLO
import supervision as sv

class GradioLineSelector:
    def __init__(self):
        self.line_start = None
        self.line_end = None
        self.frame = None
        self.video_path = None
        self.frame_index = 0
        self.total_frames = 0
        self.cap = None
    
    def load_video(self, video_path):
        """Load a video and return the first frame"""
        if video_path is None:
            return None, "No video selected"
        
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            return None, f"Error: Could not open video '{video_path}'"
        
        # Get video properties
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_index = 0
        
        # Read the first frame
        ret, self.frame = self.cap.read()
        if not ret:
            return None, "Error reading frame from video"
        
        # Reset line coordinates
        self.line_start = None
        self.line_end = None
        
        # Convert BGR to RGB for Gradio
        frame_rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        
        return frame_rgb, f"Video loaded: {os.path.basename(video_path)} ({self.total_frames} frames)"
    
    def update_frame(self, frame_slider):
        """Update the displayed frame based on slider position"""
        if self.cap is None or self.frame is None:
            return None, "No video loaded"
        
        # Convert slider value to frame index
        self.frame_index = min(int(frame_slider), self.total_frames - 1)
        
        # Set the video position
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_index)
        ret, self.frame = self.cap.read()
        
        if not ret:
            return None, f"Error reading frame {self.frame_index}"
        
        # Draw existing line if available
        frame_display = self.frame.copy()
        if self.line_start and self.line_end:
            cv2.line(frame_display, self.line_start, self.line_end, (255, 0, 255), 2)
        
        # Convert BGR to RGB for Gradio
        frame_rgb = cv2.cvtColor(frame_display, cv2.COLOR_BGR2RGB)
        
        return frame_rgb, f"Frame: {self.frame_index}/{self.total_frames}"
    
    def draw_line(self, evt: gr.SelectData):
        """Handle line drawing on the image"""
        if self.frame is None:
            return None, "No video loaded"
        
        # Get coordinates from the event
        x, y = evt.index
        
        # If this is the first point, set it as line start
        if self.line_start is None:
            self.line_start = (int(x), int(y))
            message = f"Line start set at ({x}, {y}). Click again to set end point."
        else:
            # If we already have a start point, set this as line end
            self.line_end = (int(x), int(y))
            message = f"Line set from ({self.line_start[0]}, {self.line_start[1]}) to ({x}, {y})"
        
        # Draw the line on the frame
        frame_display = self.frame.copy()
        if self.line_start:
            # Draw the start point
            cv2.circle(frame_display, self.line_start, 5, (0, 0, 255), -1)
        
        if self.line_start and self.line_end:
            # Draw the complete line
            cv2.line(frame_display, self.line_start, self.line_end, (255, 0, 255), 2)
        
        # Convert BGR to RGB for Gradio
        frame_rgb = cv2.cvtColor(frame_display, cv2.COLOR_BGR2RGB)
        
        return frame_rgb, message
    
    def reset_line(self):
        """Reset the line coordinates"""
        if self.frame is None:
            return None, "No video loaded"
        
        self.line_start = None
        self.line_end = None
        
        # Convert BGR to RGB for Gradio
        frame_rgb = cv2.cvtColor(self.frame.copy(), cv2.COLOR_BGR2RGB)
        
        return frame_rgb, "Line reset. Click on the image to draw a new line."
    
    def run_counting(self, model_type, model_size, confidence):
        """Run the people counting algorithm on the video"""
        # Ensure confidence is a Python native float, not float32
        confidence = float(confidence)
        if self.video_path is None:
            return None, "No video loaded"
        
        if self.line_start is None or self.line_end is None:
            return None, "Please draw a counting line first"
        
        # Create output directory if it doesn't exist
        output_dir = os.path.join(os.getcwd(), "output")
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate a unique filename based on timestamp
        timestamp = int(time.time())
        output_filename = f"people_counting_{timestamp}.mp4"
        output_path = os.path.join(output_dir, output_filename)
        
        # Get model path
        model_path = f"{model_type}{model_size}.pt"
        if not os.path.exists(model_path):
            return None, f"Model file {model_path} not found. Please run setup_and_demo.py first to download it."
        
        try:
            # Process the video
            result_video, stats = self.process_video(
                self.video_path, 
                output_path, 
                model_path, 
                self.line_start, 
                self.line_end, 
                float(confidence)  # Ensure it's a Python float
            )
            
            if result_video is None:
                return None, f"Error processing video: {stats}"
            
            # Check if ffprobe is available
            if shutil.which("ffprobe") is None:
                # If ffprobe is not available, we can't use Gradio's video component
                # Instead, return a message with the path to the output video
                return None, f"{stats}\n\nOutput video saved to: {output_path}\n(Note: Video preview not available because ffprobe is not installed)"
            
            return result_video, stats
            
        except Exception as e:
            error_msg = f"Error processing video: {str(e)}"
            print(error_msg)
            return None, error_msg
    
    def process_video(self, video_path, output_path, model_path, line_start, line_end, confidence):
        """Process the video and count people crossing the line"""
        # Use the process_video function from people_counter.py
        start_time = time.time()
        
        # Call the process_video function from people_counter
        result_path, frame_count, up_count, down_count = people_counter.process_video(
            video_path=video_path,
            line_start=line_start,
            line_end=line_end,
            model_path=model_path,
            confidence=float(confidence),  # Ensure it's a Python native float
            classes=[0],  # Class 0 is person in COCO dataset
            output_path=output_path,
            show=False  # Don't show in the terminal window
        )
        
        if result_path is None:
            return None, f"Error processing video"
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Generate statistics
        stats = (
            f"Processing complete. {frame_count} frames processed in {processing_time:.2f} seconds.\n"
            f"People count - Up: {up_count}, Down: {down_count}, "
            f"Total: {up_count + down_count}"
        )
        
        return result_path, stats

def create_interface():
    # Create the line selector
    line_selector = GradioLineSelector()
    
    # Define the interface
    with gr.Blocks(title="People Counting with YOLOv8") as interface:
        gr.Markdown("# People Counting with YOLOv8")
        gr.Markdown("Upload a video, select a frame, draw a counting line, and run the counter.")
        
        with gr.Row():
            with gr.Column(scale=2):
                # Video input
                video_input = gr.Video(label="Input Video", format="mp4")
                load_btn = gr.Button("Load Video")
                
                # Frame selection
                frame_slider = gr.Slider(
                    minimum=0, 
                    maximum=100, 
                    value=0, 
                    step=1, 
                    label="Frame Selection"
                )
                update_frame_btn = gr.Button("Update Frame")
                
                # Line drawing instructions
                gr.Markdown("### Drawing Instructions")
                gr.Markdown("1. Click once on the image to set the start point of the line")
                gr.Markdown("2. Click again to set the end point")
                gr.Markdown("3. Use the 'Reset Line' button to start over")
                
                # Reset line button
                reset_line_btn = gr.Button("Reset Line")
                
                # Model selection
                model_type = gr.Radio(
                    choices=["yolov8", "yolov10", "yolov11"],
                    value="yolov8",
                    label="YOLO Model Type",
                    info="Select between YOLOv8, YOLOv10, and YOLOv11"
                )
                
                model_size = gr.Radio(
                    choices=["n", "s", "m", "l", "x"],
                    value="n",
                    label="Model Size",
                    info="n=nano, s=small, m=medium, l=large, x=xlarge"
                )
                
                # Confidence threshold
                confidence = gr.Slider(
                    minimum=0.1,
                    maximum=0.9,
                    value=0.3,
                    step=0.05,
                    label="Confidence Threshold"
                )
                
                # Run button
                run_btn = gr.Button("Run People Counter", variant="primary")
            
            with gr.Column(scale=3):
                # Image display for drawing
                image_display = gr.Image(label="Video Frame", interactive=True)
                
                # Status message
                status_msg = gr.Textbox(label="Status", interactive=False)
                
                # Results
                result_video = gr.Video(label="Result Video")
                result_stats = gr.Textbox(label="Statistics", interactive=False)
        
        # Set up event handlers
        load_btn.click(
            fn=line_selector.load_video,
            inputs=[video_input],
            outputs=[image_display, status_msg]
        )
        
        update_frame_btn.click(
            fn=line_selector.update_frame,
            inputs=[frame_slider],
            outputs=[image_display, status_msg]
        )
        
        image_display.select(
            fn=line_selector.draw_line,
            inputs=[],
            outputs=[image_display, status_msg]
        )
        
        reset_line_btn.click(
            fn=line_selector.reset_line,
            inputs=[],
            outputs=[image_display, status_msg]
        )
        
        run_btn.click(
            fn=line_selector.run_counting,
            inputs=[model_type, model_size, confidence],
            outputs=[result_video, result_stats]
        )
        
        # Update frame slider max value when video is loaded
        def update_slider(video_path):
            if video_path is None:
                return gr.update(maximum=100, value=0)
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return gr.update(maximum=100, value=0)
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            return gr.update(maximum=total_frames-1, value=0)
        
        video_input.change(
            fn=update_slider,
            inputs=[video_input],
            outputs=[frame_slider]
        )
    
    return interface

if __name__ == "__main__":
    # Check if YOLO model exists
    if not (os.path.exists("yolov8n.pt") or os.path.exists("yolov10n.pt")):
        print("YOLO model not found. Running setup script to download it...")
        subprocess.run(["python", "setup_and_demo.py", "--no-demo"], check=True)
    
    # Check if ffmpeg is installed
    ffmpeg_installed = shutil.which("ffmpeg") is not None
    ffprobe_installed = shutil.which("ffprobe") is not None
    
    if not ffmpeg_installed or not ffprobe_installed:
        print("WARNING: FFmpeg or ffprobe not found in PATH.")
        print("Some video processing features may not work correctly.")
        print("Please install FFmpeg to enable all features:")
        print("  - macOS: brew install ffmpeg")
        print("  - Linux: apt-get install ffmpeg")
        print("  - Windows: Download from https://ffmpeg.org/download.html")
    
    # Create and launch the interface
    interface = create_interface()
    interface.launch()
