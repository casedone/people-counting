#!/usr/bin/env python3
import os
import cv2
import gradio as gr
import subprocess
import time
import datetime
import shutil
import people_counter
import people_detector

class GradioDetector:
    def __init__(self):
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
        
        # Convert BGR to RGB for Gradio
        frame_rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        
        return frame_rgb, f"Frame: {self.frame_index}/{self.total_frames}"
    
    def run_detection(self, model_type, model_size, confidence):
        """Run the people detection algorithm on the video"""
        # Ensure confidence is a Python native float, not float32

        confidence = float(confidence)
        if self.video_path is None:
            return None, "No video loaded"
        
        # Create output directory if it doesn't exist
        output_dir = os.path.join(os.getcwd(), "output")
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate a unique filename based on original video name and timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        original_filename = os.path.splitext(os.path.basename(self.video_path))[0]
        output_filename = f"{original_filename}_detections_{timestamp}.mp4"
        output_path = os.path.join(output_dir, output_filename)
        
        # Get model path
        model_path = f"models/{model_type}{model_size}.pt"
        if not os.path.exists(model_path):
            return None, f"Model file {model_path} not found. Please run setup_and_demo.py first to download it."
        
        try:
            # Process the video
            result_video, stats = self.process_video(
                self.video_path, 
                output_path, 
                model_path, 
                float(confidence)  # Ensure it's a Python float
            )
            
            if result_video is None:
                return None, f"Error processing video: {stats}"
            
            # Check if our Python-based ffprobe replacement exists
            ffprobe_script_path = os.path.join(os.getcwd(), "ffprobe.py")
            if os.path.exists(ffprobe_script_path):
                # If we have a Python-based ffprobe replacement, we can use it directly with the video
                return result_video, stats
            else:
                # If ffprobe is not available, we can't use Gradio's video component
                # Instead, create a custom message with a direct file path that can be opened manually
                output_rel_path = os.path.relpath(output_path, os.getcwd())
                return None, f"{stats}\n\nOutput video saved to: {output_rel_path}\n\nTo view the video, please open it with your video player."
            
            return result_video, stats
            
        except Exception as e:
            error_msg = f"Error processing video: {str(e)}"
            print(error_msg)
            return None, error_msg
    
    def process_video(self, video_path, output_path, model_path, confidence):
        """Process the video and detect people"""
        # Use the process_video function from people_detector.py
        start_time = time.time()
        
        # Call the process_video function from people_detector
        result_path, frame_count, detection_count = people_detector.process_video(
            video_path=video_path,
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
            f"Total detections: {detection_count}"
        )
        
        # Save statistics to text file with similar naming convention as the video
        stats_filename = os.path.splitext(os.path.basename(output_path))[0] + ".txt"
        stats_path = os.path.join(os.path.dirname(output_path), stats_filename)
        with open(stats_path, "w") as f:
            f.write(stats)
        
        return result_path, stats


class GradioLineCounter:
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
        
        # Generate a unique filename based on original video name and timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        original_filename = os.path.splitext(os.path.basename(self.video_path))[0]
        output_filename = f"{original_filename}_counting_{timestamp}.mp4"
        output_path = os.path.join(output_dir, output_filename)
        
        # Get model path
        model_path = f"models/{model_type}{model_size}.pt"
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

            print("DEBUG ====== ")
            print(f"result_video: {result_video}")
            print(f"stats: {stats}")
            
            if result_video is None:
                return None, f"Error processing video: {stats}"
            
            # Check if our Python-based ffprobe replacement exists
            ffprobe_script_path = os.path.join(os.getcwd(), "ffprobe.py")
            if os.path.exists(ffprobe_script_path):
                # If we have a Python-based ffprobe replacement, we can use it directly with the video
                return result_video, stats
            else:
                # If ffprobe is not available, we can't use Gradio's video component
                # Instead, create a custom message with a direct file path that can be opened manually
                output_rel_path = os.path.relpath(output_path, os.getcwd())
                return None, f"{stats}\n\nOutput video saved to: {output_rel_path}\n\nTo view the video, please open it with your video player."
            
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
        
        # Save statistics to text file with similar naming convention as the video
        stats_filename = os.path.splitext(os.path.basename(output_path))[0] + ".txt"
        stats_path = os.path.join(os.path.dirname(output_path), stats_filename)
        with open(stats_path, "w") as f:
            f.write(stats)
        
        return result_path, stats

def create_interface():
    # Create the line selector and detector
    line_counter = GradioLineCounter()
    detector = GradioDetector()
    
    # Define the interface
    with gr.Blocks(title="People Analysis with YOLO") as interface:
        gr.Markdown("# People Analysis with YOLO")
        
        # Create tabs
        with gr.Tabs() as tabs:
            # People Counting Tab
            with gr.TabItem("People Counting"):
                gr.Markdown("Upload a video, select a frame, draw a counting line, and run the counter.")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        # Video input
                        count_video_input = gr.Video(label="Input Video", format="mp4")
                        count_load_btn = gr.Button("Load Video")
                        
                        # Frame selection
                        count_frame_slider = gr.Slider(
                            minimum=0, 
                            maximum=100, 
                            value=0, 
                            step=1, 
                            label="Frame Selection"
                        )
                        count_update_frame_btn = gr.Button("Update Frame")
                        
                        # Line drawing instructions
                        gr.Markdown("### Drawing Instructions")
                        gr.Markdown("1. Click once on the image to set the start point of the line")
                        gr.Markdown("2. Click again to set the end point")
                        gr.Markdown("3. Use the 'Reset Line' button to start over")
                        
                        # Reset line button
                        count_reset_line_btn = gr.Button("Reset Line")
                        
                        # Model selection
                        count_model_type = gr.Radio(
                            choices=["yolo12"],
                            value="yolo12",
                            label="YOLO Model Type",
                            info="Using YOLO12 for object detection"
                        )
                        
                        count_model_size = gr.Radio(
                            choices=["n", "s", "m", "l", "x"],
                            value="n",
                            label="Model Size",
                            info="n=nano, s=small, m=medium, l=large, x=xlarge"
                        )
                        
                        # Confidence threshold
                        count_confidence = gr.Slider(
                            minimum=0.1,
                            maximum=0.9,
                            value=0.3,
                            step=0.05,
                            label="Confidence Threshold"
                        )
                        
                        # Run button
                        count_run_btn = gr.Button("Run People Counter", variant="primary")
                    
                    with gr.Column(scale=3):
                        # Image display for drawing
                        count_image_display = gr.Image(label="Video Frame", interactive=True)
                        
                        # Status message
                        count_status_msg = gr.Textbox(label="Status", interactive=False)
                        
                        # Results
                        count_result_video = gr.Video(label="Result Video")
                        count_result_stats = gr.Textbox(label="Statistics", interactive=False)
            
            # People Detection Tab
            with gr.TabItem("People Detection"):
                gr.Markdown("Upload a video, select a frame, and run the detector to see bounding boxes.")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        # Video input
                        detect_video_input = gr.Video(label="Input Video", format="mp4")
                        detect_load_btn = gr.Button("Load Video")
                        
                        # Frame selection
                        detect_frame_slider = gr.Slider(
                            minimum=0, 
                            maximum=100, 
                            value=0, 
                            step=1, 
                            label="Frame Selection"
                        )
                        detect_update_frame_btn = gr.Button("Update Frame")
                        
                        # Model selection
                        detect_model_type = gr.Radio(
                            choices=["yolo12"],
                            value="yolo12",
                            label="YOLO Model Type",
                            info="Using YOLO12 for object detection"
                        )
                        
                        detect_model_size = gr.Radio(
                            choices=["n", "s", "m", "l", "x"],
                            value="n",
                            label="Model Size",
                            info="n=nano, s=small, m=medium, l=large, x=xlarge"
                        )
                        
                        # Confidence threshold
                        detect_confidence = gr.Slider(
                            minimum=0.1,
                            maximum=0.9,
                            value=0.3,
                            step=0.05,
                            label="Confidence Threshold"
                        )
                        
                        # Run button
                        detect_run_btn = gr.Button("Run People Detector", variant="primary")
                    
                    with gr.Column(scale=3):
                        # Image display
                        detect_image_display = gr.Image(label="Video Frame")
                        
                        # Status message
                        detect_status_msg = gr.Textbox(label="Status", interactive=False)
                        
                        # Results
                        detect_result_video = gr.Video(label="Result Video")
                        detect_result_stats = gr.Textbox(label="Statistics", interactive=False)
        
        # Set up event handlers for People Counting tab
        count_load_btn.click(
            fn=line_counter.load_video,
            inputs=[count_video_input],
            outputs=[count_image_display, count_status_msg]
        )
        
        count_update_frame_btn.click(
            fn=line_counter.update_frame,
            inputs=[count_frame_slider],
            outputs=[count_image_display, count_status_msg]
        )
        
        count_image_display.select(
            fn=line_counter.draw_line,
            inputs=[],
            outputs=[count_image_display, count_status_msg]
        )
        
        count_reset_line_btn.click(
            fn=line_counter.reset_line,
            inputs=[],
            outputs=[count_image_display, count_status_msg]
        )
        
        count_run_btn.click(
            fn=line_counter.run_counting,
            inputs=[count_model_type, count_model_size, count_confidence],
            outputs=[count_result_video, count_result_stats]
        )
        
        # Set up event handlers for People Detection tab
        detect_load_btn.click(
            fn=detector.load_video,
            inputs=[detect_video_input],
            outputs=[detect_image_display, detect_status_msg]
        )
        
        detect_update_frame_btn.click(
            fn=detector.update_frame,
            inputs=[detect_frame_slider],
            outputs=[detect_image_display, detect_status_msg]
        )
        
        detect_run_btn.click(
            fn=detector.run_detection,
            inputs=[detect_model_type, detect_model_size, detect_confidence],
            outputs=[detect_result_video, detect_result_stats]
        )
        
        # Update frame slider max value when videos are loaded
        def update_slider(video_path):
            if video_path is None:
                return gr.update(maximum=100, value=0)
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return gr.update(maximum=100, value=0)
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            return gr.update(maximum=total_frames-1, value=0)
        
        count_video_input.change(
            fn=update_slider,
            inputs=[count_video_input],
            outputs=[count_frame_slider]
        )
        
        detect_video_input.change(
            fn=update_slider,
            inputs=[detect_video_input],
            outputs=[detect_frame_slider]
        )
    
    return interface

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description="People Counting and Detection Interface")
    parser.add_argument("--share", action="store_true", default=False, 
                        help="Whether to create a publicly shareable link (default: True)")
    args = parser.parse_args()
    
    # Check if YOLO model exists
    if not os.path.exists("models/yolo12n.pt"):
        print("YOLO12 model not found. Running setup script to download it...")
        subprocess.run(["python", "setup_and_demo.py", "--no-demo"], check=True)
    
    # Set up local ffmpeg and ffprobe binaries
    try:
        from people_counter import setup_local_ffmpeg
        ffmpeg_path, ffprobe_path = setup_local_ffmpeg()
        print("Successfully configured local ffmpeg and ffprobe binaries")
    except Exception as e:
        print(f"Warning: Failed to set up local ffmpeg/ffprobe: {e}")
        print("Falling back to system-installed ffmpeg/ffprobe if available")
        
        # Try to use the old method as fallback
        ffmpeg_dir = os.path.join(os.getcwd(), "ffmpeg")
        ffmpeg_path = os.path.join(ffmpeg_dir, "ffmpeg")
        ffprobe_script_path = os.path.join(os.getcwd(), "ffprobe.py")
        
        # Add current directory to PATH to find the ffprobe script
        os.environ["PATH"] = os.getcwd() + os.pathsep + os.environ["PATH"]
        
        # Check if ffmpeg executables exist in the local directory
        if os.path.exists(ffmpeg_dir):
            # Add ffmpeg directory to PATH environment variable
            os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ["PATH"]
            print(f"Using local ffmpeg installation from: {ffmpeg_dir}")
        
        # Check if Python-based ffprobe replacement exists
        if os.path.exists(ffprobe_script_path):
            print(f"Using Python-based ffprobe replacement: {ffprobe_script_path}")
            # Set environment variables for ffprobe path that Gradio might use
            os.environ["FFPROBE_PATH"] = ffprobe_script_path
            os.environ["GRADIO_FFPROBE_PATH"] = ffprobe_script_path
    
    # Check if ffmpeg is installed (either locally or system-wide)
    ffmpeg_installed = shutil.which("ffmpeg") is not None
    
    if not ffmpeg_installed:
        print("WARNING: FFmpeg not found in PATH or local directory.")
        print("Some video processing features may not work correctly.")
    
    # Create and launch the interface
    interface = create_interface()
    interface.launch(share=args.share)
