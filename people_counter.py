import argparse
import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
from collections import defaultdict
import os
import datetime
from tqdm import tqdm

class LineCounter:
    def __init__(self, start_point, end_point, counting_region=15):
        """
        Initialize a line counter for tracking objects crossing a line.
        
        Args:
            start_point: Starting point of the line (x1, y1)
            end_point: Ending point of the line (x2, y2)
            counting_region: Width of the region around the line to detect crossings
        """
        self.start_point = np.array(start_point)
        self.end_point = np.array(end_point)
        self.counting_region = counting_region
        
        # Calculate line properties
        self.line_vector = self.end_point - self.start_point
        self.line_length = np.linalg.norm(self.line_vector)
        self.unit_line_vector = self.line_vector / self.line_length
        self.normal_vector = np.array([-self.unit_line_vector[1], self.unit_line_vector[0]])
        
        # Tracking data
        self.object_tracks = defaultdict(list)
        self.crossed_objects = set()
        self.up_count = 0
        self.down_count = 0
    
    def get_distance_from_line(self, point):
        """Calculate the signed distance from a point to the line."""
        point = np.array(point)
        return np.dot(point - self.start_point, self.normal_vector)
    
    def is_point_in_counting_region(self, point):
        """Check if a point is within the counting region around the line."""
        # Project the point onto the line
        point = np.array(point)
        point_vector = point - self.start_point
        projection_length = np.dot(point_vector, self.unit_line_vector)
        
        # Check if projection is within line segment
        if projection_length < 0 or projection_length > self.line_length:
            return False
        
        # Check distance from line
        distance = abs(self.get_distance_from_line(point))
        return distance <= self.counting_region
    
    def update(self, object_id, center_point):
        """
        Update tracking for an object and check if it crossed the line.
        
        Args:
            object_id: Unique identifier for the object
            center_point: Current center point of the object (x, y)
            
        Returns:
            True if the object crossed the line in this update, False otherwise
        """
        if object_id in self.crossed_objects:
            return False
        
        # Add current position to track
        self.object_tracks[object_id].append(center_point)
        
        # Need at least 2 points to detect crossing
        if len(self.object_tracks[object_id]) < 2:
            return False
        
        # Get current and previous positions
        current_pos = np.array(self.object_tracks[object_id][-1])
        prev_pos = np.array(self.object_tracks[object_id][-2])
        
        # Calculate distances from line
        current_distance = self.get_distance_from_line(current_pos)
        prev_distance = self.get_distance_from_line(prev_pos)
        
        # Check if the object crossed the line (sign change in distance)
        if current_distance * prev_distance <= 0 and abs(current_distance) <= self.counting_region:
            self.crossed_objects.add(object_id)
            
            # Determine direction of crossing
            if current_distance > prev_distance:
                self.up_count += 1
                return "up"
            else:
                self.down_count += 1
                return "down"
        
        return False

def parse_arguments():
    parser = argparse.ArgumentParser(description="Count people crossing a line in a video using YOLO12")
    parser.add_argument("--video", type=str, required=True, help="Path to input video file")
    parser.add_argument("--model", type=str, default="models/yolo12n.pt", help="Path to YOLO model")
    parser.add_argument("--model-type", type=str, choices=["yolo12"], default="yolo12", 
                        help="Type of YOLO model to use (yolo12)")
    parser.add_argument("--line-start", type=int, nargs=2, default=[0, 0], help="Starting point of counting line (x y)")
    parser.add_argument("--line-end", type=int, nargs=2, default=[0, 0], help="Ending point of counting line (x y)")
    parser.add_argument("--confidence", type=float, default=0.3, help="Detection confidence threshold")
    parser.add_argument("--output", type=str, default="", help="Path to output video file")
    parser.add_argument("--show", action="store_true", help="Display the video while processing")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--classes", type=int, nargs="+", default=[0], help="Classes to detect (default: 0 for person)")
    return parser.parse_args()

def process_video(video_path, line_start, line_end, model_path, confidence=0.3, classes=[0], 
                 output_path="object_counting_output.mp4", show=False, verbose=False):
    """
    Process a video to count people crossing a line with progress tracking.
    
    Args:
        video_path: Path to the input video
        line_start: Starting point of the counting line (x, y)
        line_end: Ending point of the counting line (x, y)
        model_path: Path to the YOLO model
        confidence: Detection confidence threshold
        classes: List of classes to detect
        output_path: Path to save the output video
        show: Whether to display the video while processing
        verbose: Whether to enable verbose output for the YOLO model
        
    Note:
        Displays a progress bar showing frames processed, percentage complete, and estimated time remaining.
        
    Returns:
        tuple: (output_path, frame_count, up_count, down_count)
    """
    # Check if video exists
    if not os.path.isfile(video_path):
        print(f"Error: Video file '{video_path}' not found")
        return None, 0, 0, 0
    
    # Load the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video '{video_path}'")
        return None, 0, 0, 0
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Set default line if not provided
    if line_start == [0, 0] and line_end == [0, 0]:
        # Default to a horizontal line in the middle of the frame
        line_start = [0, frame_height // 2]
        line_end = [frame_width, frame_height // 2]
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    frame_count = 0
    up_count = 0
    down_count = 0
    
    # Use the LineCounter approach
    line_counter = LineCounter(line_start, line_end)
    
    # Initialize YOLO model
    print(f"model_path = {model_path}")
    model = YOLO(model_path, task='detect', verbose=verbose)
    
    # Initialize tracker
    tracker = sv.ByteTrack()
    
    # Get total frame count for progress bar
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Process the video with progress bar
    progress_bar = tqdm(total=total_frames, desc="Processing video", 
                        unit="frames", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        progress_bar.update(1)
        
        # Run YOLO inference on the frame
        # Ensure confidence is a Python native float, not float32
        results = model(frame, conf=float(confidence), classes=classes, verbose=False)
        
        # Get detections
        detections = sv.Detections.from_ultralytics(results[0])
        
        # Update tracker
        detections = tracker.update_with_detections(detections)
        
        # Process each detection
        for i, (xyxy, _confidence, class_id, tracker_id) in enumerate(zip(
            detections.xyxy, detections.confidence, detections.class_id, detections.tracker_id
        )):
            if tracker_id is None:
                continue
                
            # Calculate center point of the bounding box
            x1, y1, x2, y2 = xyxy
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # Update line counter
            crossing = line_counter.update(tracker_id, (center_x, center_y))
            
            # Draw bounding box
            color = (0, 255, 0)  # Green for default
            if crossing == "up":
                color = (0, 0, 255)  # Red for up crossing
            elif crossing == "down":
                color = (255, 0, 0)  # Blue for down crossing
                
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            
            # Draw ID
            cv2.putText(frame, f"ID: {tracker_id}", (int(x1), int(y1) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw the counting line
        cv2.line(frame, tuple(line_start), tuple(line_end), (255, 0, 255), 2)
        
        # Draw counting region
        region_points = []
        for t in np.linspace(0, 1, 100):
            point = np.array(line_start) + t * (np.array(line_end) - np.array(line_start))
            region_points.append(point + line_counter.normal_vector * line_counter.counting_region)
        
        for t in np.linspace(1, 0, 100):
            point = np.array(line_start) + t * (np.array(line_end) - np.array(line_start))
            region_points.append(point - line_counter.normal_vector * line_counter.counting_region)
        
        region_points = np.array(region_points, dtype=np.int32)
        cv2.polylines(frame, [region_points], True, (255, 0, 255), 1)
        
        # Draw counts
        cv2.putText(frame, f"Up: {line_counter.up_count} Down: {line_counter.down_count}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Display the frame
        if show:
            cv2.imshow('People Counter', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Write the frame to output video
        output_writer.write(frame)
    
    # Close progress bar
    progress_bar.close()
    
    # Get counts
    up_count = line_counter.up_count
    down_count = line_counter.down_count
    
    # Release resources
    cap.release()
    output_writer.release()
    cv2.destroyAllWindows()
    
    # Return results
    return output_path, frame_count, up_count, down_count

def main():
    args = parse_arguments()
    
    # Generate output filename with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    original_filename = os.path.splitext(os.path.basename(args.video))[0]
    output_filename = f"{original_filename}_counting_{timestamp}.mp4"
    
    # Ensure output directory exists
    os.makedirs("output", exist_ok=True)
    
    # Set output path
    output_path_arg = args.output if args.output else os.path.join("output", output_filename)
    
    # Process the video
    output_path, frame_count, up_count, down_count = process_video(
        video_path=args.video,
        line_start=args.line_start,
        line_end=args.line_end,
        model_path=args.model,
        confidence=args.confidence,
        classes=args.classes,
        output_path=output_path_arg,
        show=args.show,
        verbose=args.verbose
    )
    
    # Print results
    if output_path:
        print(f"Processing complete. {frame_count} frames processed.")
        print(f"People count - Up: {up_count}, Down: {down_count}")

if __name__ == "__main__":
    main()
