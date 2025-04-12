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
import folder_cleaner

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
            # Folder Management Tab
            with gr.TabItem("Folder Management"):
                gr.Markdown("# Folder Management")
                gr.Markdown("Clear input, output, and batch_jobs folders individually or in any combination.")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        # Folder selection
                        gr.Markdown("### Select folders to clear")
                        clear_input = gr.Checkbox(label="Clear input folder", value=False)
                        clear_output = gr.Checkbox(label="Clear output folder", value=False)
                        clear_batch_jobs = gr.Checkbox(label="Clear batch_jobs folder", value=False)
                        clear_all = gr.Checkbox(label="Clear all folders", value=False)
                        
                        # Options
                        gr.Markdown("### Options")
                        delete_gitkeep = gr.Checkbox(label="Delete .gitkeep files", value=False, 
                                                    info="By default, .gitkeep files are preserved")
                        
                        # Clear button
                        clear_btn = gr.Button("Clear Selected Folders", variant="primary")
                    
                    with gr.Column(scale=2):
                        # Status message
                        clear_status = gr.Textbox(label="Status", interactive=False)
                        
                        # Folder contents
                        gr.Markdown("### Current folder contents")
                        folder_contents = gr.Textbox(label="Folder Contents", interactive=False, lines=15)
                        refresh_btn = gr.Button("Refresh Folder Contents")
                
                # Function to clear folders
                def clear_folders(clear_input, clear_output, clear_batch_jobs, clear_all, delete_gitkeep):
                    keep_gitkeep = not delete_gitkeep
                    result = []
                    
                    try:
                        if clear_all:
                            count = folder_cleaner.clear_all_folders(keep_gitkeep)
                            result.append(f"Cleared all folders: {count} files/folders removed")
                        else:
                            folders_to_clear = []
                            if clear_input:
                                folders_to_clear.append('input')
                            if clear_output:
                                folders_to_clear.append('output')
                            if clear_batch_jobs:
                                folders_to_clear.append('batch_jobs')
                            
                            if not folders_to_clear:
                                return "No folders selected for clearing"
                            
                            count = folder_cleaner.clear_folders(folders_to_clear, keep_gitkeep)
                            result.append(f"Cleared selected folders: {count} files/folders removed")
                        
                        return "\n".join(result)
                    except Exception as e:
                        return f"Error clearing folders: {str(e)}"
                
                # Function to get folder contents
                def get_folder_contents():
                    result = []
                    
                    # Check input folder
                    input_files = os.listdir("input") if os.path.exists("input") else []
                    result.append(f"Input folder ({len(input_files)} files):")
                    for file in input_files[:10]:  # Limit to first 10 files
                        result.append(f"  - {file}")
                    if len(input_files) > 10:
                        result.append(f"  - ... and {len(input_files) - 10} more files")
                    result.append("")
                    
                    # Check output folder
                    output_files = os.listdir("output") if os.path.exists("output") else []
                    result.append(f"Output folder ({len(output_files)} files):")
                    for file in output_files[:10]:  # Limit to first 10 files
                        result.append(f"  - {file}")
                    if len(output_files) > 10:
                        result.append(f"  - ... and {len(output_files) - 10} more files")
                    result.append("")
                    
                    # Check batch_jobs folder
                    batch_jobs_files = os.listdir("batch_jobs") if os.path.exists("batch_jobs") else []
                    result.append(f"Batch jobs folder ({len(batch_jobs_files)} files):")
                    for file in batch_jobs_files[:10]:  # Limit to first 10 files
                        result.append(f"  - {file}")
                    if len(batch_jobs_files) > 10:
                        result.append(f"  - ... and {len(batch_jobs_files) - 10} more files")
                    
                    return "\n".join(result)
                
                # Set up event handlers for Folder Management tab
                clear_btn.click(
                    fn=clear_folders,
                    inputs=[clear_input, clear_output, clear_batch_jobs, clear_all, delete_gitkeep],
                    outputs=[clear_status]
                )
                
                refresh_btn.click(
                    fn=get_folder_contents,
                    inputs=[],
                    outputs=[folder_contents]
                )
                
                # Auto-update folder contents on page load
                # Using gr.on to set initial value instead of .update() which is no longer supported
                interface.load(
                    fn=get_folder_contents,
                    inputs=[],
                    outputs=[folder_contents]
                )
                
                # Update clear_all checkbox to control other checkboxes
                def update_checkboxes(clear_all_value):
                    if clear_all_value:
                        return gr.update(value=True, interactive=False), gr.update(value=True, interactive=False), gr.update(value=True, interactive=False)
                    else:
                        return gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True)
                
                clear_all.change(
                    fn=update_checkboxes,
                    inputs=[clear_all],
                    outputs=[clear_input, clear_output, clear_batch_jobs]
                )
            
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
    # First check if we have the local binaries
    bin_ffmpeg_path = os.path.join(os.getcwd(), "bin", "ffmpeg")
    if os.path.exists(bin_ffmpeg_path) and os.access(bin_ffmpeg_path, os.X_OK):
        ffmpeg_installed = True
    else:
        # Fall back to checking system PATH
        ffmpeg_installed = shutil.which("ffmpeg") is not None
    
    if not ffmpeg_installed:
        print("WARNING: FFmpeg not found in PATH or local directory.")
        print("Some video processing features may not work correctly.")
    
    # Create and launch the interface
    interface = create_interface()
    interface.launch(share=args.share)
