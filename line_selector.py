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
    
    def generate_command(self):
        """Generate the command to run people_counter.py with the selected line."""
        if self.line_start and self.line_end:
            return (f"python people_counter.py --video {self.video_path} "
                   f"--line-start {self.line_start[0]} {self.line_start[1]} "
                   f"--line-end {self.line_end[0]} {self.line_end[1]} "
                   f"--show")
        return None

def main():
    parser = argparse.ArgumentParser(description="Interactive tool to select a counting line for people_counter.py")
    parser.add_argument("--video", type=str, required=True, help="Path to input video file")
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
        
        # Generate and display command
        command = selector.generate_command()
        print("\nRun the following command to count people crossing this line:")
        print(command)
        
        # Ask if user wants to run the command now
        response = input("\nDo you want to run this command now? (y/n): ")
        if response.lower() == 'y':
            import subprocess
            subprocess.run(command, shell=True)
    else:
        print("No line was selected")

if __name__ == "__main__":
    main()
