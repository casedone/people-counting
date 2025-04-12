#!/usr/bin/env python3
import cv2
import numpy as np
import argparse
import os

class LineSelector:
    def __init__(self, video_path):
        """Initialize the line selector with a video path."""
        self.video_path = video_path
        self.frame = None
        self.line_start = None
        self.line_end = None
        self.drawing = False
        self.window_name = "Line Selector - Draw a line and press Enter"
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for line drawing."""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Start drawing line
            self.drawing = True
            self.line_start = (x, y)
            self.line_end = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            # Update line end point while drawing
            self.line_end = (x, y)
            # Create a copy of the original frame to draw on
            img_copy = self.frame.copy()
            cv2.line(img_copy, self.line_start, self.line_end, (255, 0, 255), 2)
            cv2.imshow(self.window_name, img_copy)
        elif event == cv2.EVENT_LBUTTONUP:
            # Finish drawing line
            self.drawing = False
            self.line_end = (x, y)
            # Draw the final line
            cv2.line(self.frame, self.line_start, self.line_end, (255, 0, 255), 2)
            cv2.imshow(self.window_name, self.frame)
    
    def select_frame(self):
        """Allow user to select a frame from the video."""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video '{self.video_path}'")
            return False
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Default to a frame 1/3 of the way through the video
        target_frame = int(total_frames / 3)
        
        # Create a trackbar window
        cv2.namedWindow("Frame Selector")
        cv2.createTrackbar("Frame", "Frame Selector", target_frame, total_frames - 1, lambda x: None)
        
        while True:
            # Get the current position from the trackbar
            pos = cv2.getTrackbarPos("Frame", "Frame Selector")
            
            # Set the video position
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ret, frame = cap.read()
            
            if not ret:
                print("Error reading frame")
                break
            
            # Display frame info
            time_sec = pos / fps
            minutes = int(time_sec / 60)
            seconds = int(time_sec % 60)
            info_text = f"Frame: {pos}/{total_frames} - Time: {minutes:02d}:{seconds:02d}"
            
            # Add text to the frame
            cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "Use trackbar to select a frame, then press Enter", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Show the frame
            cv2.imshow("Frame Selector", frame)
            
            # Wait for key press
            key = cv2.waitKey(1) & 0xFF
            if key == 13:  # Enter key
                self.frame = frame.copy()
                cv2.destroyWindow("Frame Selector")
                return True
            elif key == 27:  # Escape key
                cv2.destroyWindow("Frame Selector")
                return False
        
        cap.release()
        return False
    
    def draw_line(self):
        """Allow user to draw a line on the selected frame."""
        if self.frame is None:
            print("No frame selected")
            return False
        
        # Create window and set mouse callback
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        # Display instructions
        cv2.putText(self.frame, "Click and drag to draw a line", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(self.frame, "Press Enter to confirm, Escape to cancel", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow(self.window_name, self.frame)
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 13:  # Enter key
                if self.line_start and self.line_end:
                    cv2.destroyWindow(self.window_name)
                    return True
            elif key == 27:  # Escape key
                cv2.destroyWindow(self.window_name)
                return False
        
        return False
    
    def get_line_coordinates(self):
        """Return the line coordinates."""
        if self.line_start and self.line_end:
            return self.line_start, self.line_end
        return None, None
    
    def generate_command(self, model_type="yolov8"):
        """Generate the command to run people_counter.py with the selected line."""
        if self.line_start and self.line_end:
            return (f"python people_counter.py --video {self.video_path} "
                   f"--line-start {self.line_start[0]} {self.line_start[1]} "
                   f"--line-end {self.line_end[0]} {self.line_end[1]} "
                   f"--model-type {model_type} "
                   f"--show")
        return None
    
    def process_video_directly(self, model_type="yolov8", model_size="n"):
        """Process the video directly using the process_video function from people_counter.py."""
        if not self.line_start or not self.line_end:
            print("No line selected")
            return False
        
        try:
            import people_counter
            
            # Set up local ffmpeg and ffprobe binaries
            try:
                ffmpeg_path, ffprobe_path = people_counter.setup_local_ffmpeg()
                print("Successfully configured local ffmpeg and ffprobe binaries")
            except Exception as e:
                print(f"Warning: Failed to set up local ffmpeg/ffprobe: {e}")
                print("Falling back to system-installed ffmpeg/ffprobe if available")
            
            # Generate output path
            import time
            output_dir = os.path.join(os.getcwd(), "output")
            os.makedirs(output_dir, exist_ok=True)
            timestamp = int(time.time())
            output_path = os.path.join(output_dir, f"people_counting_{timestamp}.mp4")
            
            # Get model path
            model_path = f"{model_type}{model_size}.pt"
            if not os.path.exists(model_path):
                print(f"Model file {model_path} not found")
                return False
            
            print(f"Processing video: {self.video_path}")
            print(f"Line: from {self.line_start} to {self.line_end}")
            print(f"Model: {model_path}")
            print(f"Output: {output_path}")
            
            # Process the video
            result_path, frame_count, up_count, down_count = people_counter.process_video(
                video_path=self.video_path,
                line_start=self.line_start,
                line_end=self.line_end,
                model_path=model_path,
                confidence=0.3,
                classes=[0],  # Class 0 is person in COCO dataset
                output_path=output_path,
                show=True  # Show the video while processing
            )
            
            if result_path:
                print(f"\nProcessing complete. {frame_count} frames processed.")
                print(f"People count - Up: {up_count}, Down: {down_count}, Total: {up_count + down_count}")
                print(f"Output video saved to: {output_path}")
                return True
            else:
                print("Error processing video")
                return False
                
        except ImportError:
            print("Could not import people_counter module. Using command line approach instead.")
            command = self.generate_command(model_type)
            if command:
                import subprocess
                subprocess.run(command, shell=True)
                return True
            return False

def main():
    parser = argparse.ArgumentParser(description="Interactive tool to select a counting line for people_counter.py")
    parser.add_argument("--video", type=str, required=True, help="Path to input video file")
    parser.add_argument("--model-type", type=str, choices=["yolo12"], default="yolo12",
                        help="Type of YOLO model to use (yolo12)")
    parser.add_argument("--model-size", type=str, choices=["n", "s", "m", "l", "x"], default="n",
                        help="YOLO model size: n(ano), s(mall), m(edium), l(arge), x(large)")
    args = parser.parse_args()
    
    # Check if video exists
    if not os.path.isfile(args.video):
        print(f"Error: Video file '{args.video}' not found")
        return
    
    # Create line selector
    selector = LineSelector(args.video)
    
    # Select a frame
    print("Select a frame from the video...")
    if not selector.select_frame():
        print("Frame selection canceled")
        return
    
    # Draw a line
    print("Draw a line on the frame...")
    if not selector.draw_line():
        print("Line drawing canceled")
        return
    
    # Get line coordinates
    start, end = selector.get_line_coordinates()
    if start and end:
        print(f"Line selected: from {start} to {end}")
        
        # Generate command for display purposes
        command = selector.generate_command(args.model_type)
        print("\nEquivalent command line:")
        print(command)
        
        # Ask if user wants to process the video now
        response = input("\nDo you want to process the video now? (y/n): ")
        if response.lower() == 'y':
            # Try to use the direct processing method first
            try:
                selector.process_video_directly(
                    model_type=args.model_type,
                    model_size=args.model_size
                )
            except Exception as e:
                print(f"Error using direct processing: {str(e)}")
                print("Falling back to command line approach...")
                import subprocess
                subprocess.run(command, shell=True)
    else:
        print("No line was selected")

if __name__ == "__main__":
    main()
