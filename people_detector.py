import argparse
import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
import os
import datetime
from tqdm import tqdm
import sys

def parse_arguments():
    parser = argparse.ArgumentParser(description="Detect people in a video using YOLO12")
    parser.add_argument("--video", type=str, required=True, help="Path to input video file")
    parser.add_argument("--model", type=str, default="models/yolo12n.pt", help="Path to YOLO model")
    parser.add_argument("--model-type", type=str, choices=["yolo12"], default="yolo12", 
                        help="Type of YOLO model to use (yolo12)")
    parser.add_argument("--confidence", type=float, default=0.3, help="Detection confidence threshold")
    parser.add_argument("--output", type=str, default="", help="Path to output video file")
    parser.add_argument("--output-height", type=int, default=480, 
                        help="Output video height in pixels (default: 480, 0 for original resolution)")
    parser.add_argument("--show", action="store_true", help="Display the video while processing")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--classes", type=int, nargs="+", default=[0], help="Classes to detect (default: 0 for person)")
    return parser.parse_args()

def process_video(video_path, model_path, confidence=0.3, classes=[0], 
                 output_path="people_detection_output.mp4", show=False, verbose=False, output_height=480):
    """
    Process a video to detect people and draw bounding boxes with progress tracking.
    
    Args:
        video_path: Path to the input video
        model_path: Path to the YOLO model
        confidence: Detection confidence threshold
        classes: List of classes to detect
        output_path: Path to save the output video
        show: Whether to display the video while processing
        verbose: Whether to enable verbose output for the YOLO model
        output_height: Height of the output video in pixels (default: 480)
                      Width is calculated to maintain the original aspect ratio.
                      If set to 0, the original video resolution will be used.
        
    Note:
        Displays a progress bar showing frames processed, percentage complete, and estimated time remaining.
        
    Returns:
        tuple: (output_path, frame_count, detection_count)
    """
    # Check if video exists
    if not os.path.isfile(video_path):
        print(f"Error: Video file '{video_path}' not found")
        return None, 0, 0
    
    # Load the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video '{video_path}'")
        return None, 0, 0
    
    # Get video properties
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Handle special cases for output_height
    if output_height < 0:
        print(f"Warning: Negative output height ({output_height}) is invalid. Using default (480).")
        output_height = 480
    
    # Handle case where output_height is 0 (use original resolution)
    if output_height == 0 or output_height == orig_height:
        output_height = orig_height
        output_width = orig_width
        scale_x = 1.0
        scale_y = 1.0
    else:
        # Calculate output dimensions while maintaining aspect ratio
        output_width = int(orig_width * (output_height / orig_height))
        
        # Calculate scale factors for coordinate conversion
        scale_x = output_width / orig_width
        scale_y = output_height / orig_height
    
    # Initialize video writer with output dimensions
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_writer = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))
    
    # Print resolution information
    print(f"Input resolution: {orig_width}x{orig_height}")
    print(f"Output resolution: {output_width}x{output_height}")
    
    frame_count = 0
    total_detections = 0
    
    # Initialize YOLO model
    model = YOLO(model_path, task='detect', verbose=verbose)
    
    # Get total frame count for progress bar
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Process the video with progress bar
    progress_bar = tqdm(total=total_frames, desc="Processing video", 
                        unit="frames", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")
    
    # Process the video
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
        
        # Count detections in this frame
        frame_detections = len(detections)
        total_detections += frame_detections
        
        # Create a clean frame for display and a clean frame for output
        display_frame = frame.copy()
        
        # Store detection info for drawing later
        detection_info = []
        
        # Process each detection
        for i, (xyxy, _confidence, class_id) in enumerate(zip(
            detections.xyxy, detections.confidence, detections.class_id
        )):
            # Store detection info
            detection_info.append((xyxy, _confidence, class_id))
        
        # Draw on display frame (original resolution)
        for xyxy, _confidence, class_id in detection_info:
            x1, y1, x2, y2 = xyxy
            # Draw bounding box
            cv2.rectangle(display_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            # Draw class and confidence
            label = f"{model.model.names[class_id]}: {_confidence:.2f}"
            cv2.putText(display_frame, label, (int(x1), int(y1) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw detection count on display frame
        cv2.putText(display_frame, f"Detections: {frame_detections}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Create output frame with appropriate resolution
        if output_height != orig_height:
            # Start with a clean resized frame
            output_frame = cv2.resize(frame.copy(), (output_width, output_height))
            
            # Draw detections on output frame with scaled coordinates
            for xyxy, _confidence, class_id in detection_info:
                # Scale bounding box coordinates
                x1, y1, x2, y2 = xyxy
                scaled_x1 = int(x1 * scale_x)
                scaled_y1 = int(y1 * scale_y)
                scaled_x2 = int(x2 * scale_x)
                scaled_y2 = int(y2 * scale_y)
                
                # Draw bounding box with scaled coordinates
                cv2.rectangle(output_frame, (scaled_x1, scaled_y1), (scaled_x2, scaled_y2), (0, 255, 0), 2)
                
                # Draw class and confidence with scaled coordinates
                label = f"{model.model.names[class_id]}: {_confidence:.2f}"
                cv2.putText(output_frame, label, (scaled_x1, scaled_y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale_y, (0, 255, 0), max(1, int(2 * scale_y)))
            
            # Draw detection count with scaled font size and position
            cv2.putText(output_frame, f"Detections: {frame_detections}", 
                        (10, int(30 * scale_y)), cv2.FONT_HERSHEY_SIMPLEX, 
                        1 * scale_y, (0, 0, 255), max(1, int(2 * scale_y)))
        else:
            # If no resizing is needed, use the display frame
            output_frame = display_frame.copy()
        
        # Display the frame (original size for display)
        if show:
            cv2.imshow('People Detector', display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Write the output frame to video
        output_writer.write(output_frame)
    
    # Close progress bar
    progress_bar.close()
    
    # Release resources
    cap.release()
    output_writer.release()
    cv2.destroyAllWindows()
    
    # Return results
    return output_path, frame_count, total_detections

def main():
    # Set up local ffmpeg and ffprobe binaries
    try:
        from people_counter import setup_local_ffmpeg
        ffmpeg_path, ffprobe_path = setup_local_ffmpeg()
        print("Successfully configured local ffmpeg and ffprobe binaries")
    except Exception as e:
        print(f"Warning: Failed to set up local ffmpeg/ffprobe: {e}")
        print("Falling back to system-installed ffmpeg/ffprobe if available")
    
    args = parse_arguments()
    
    # Ensure output directory exists
    os.makedirs("output", exist_ok=True)
    
    # Generate output filename with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    original_filename = os.path.splitext(os.path.basename(args.video))[0]
    output_filename = f"{original_filename}_detections_{timestamp}.mp4"
    output_path = args.output if args.output else os.path.join("output", output_filename)
    
    # Process the video
    output_path, frame_count, detection_count = process_video(
        video_path=args.video,
        model_path=args.model,
        confidence=args.confidence,
        classes=args.classes,
        output_path=output_path,
        show=args.show,
        verbose=args.verbose,
        output_height=args.output_height
    )
    
    # Print results
    if output_path:
        print(f"Processing complete. {frame_count} frames processed.")
        print(f"Total detections: {detection_count}")

if __name__ == "__main__":
    main()
