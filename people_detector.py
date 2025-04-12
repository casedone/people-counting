import argparse
import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
import os
import datetime
from tqdm import tqdm

def parse_arguments():
    parser = argparse.ArgumentParser(description="Detect people in a video using YOLO12")
    parser.add_argument("--video", type=str, required=True, help="Path to input video file")
    parser.add_argument("--model", type=str, default="models/yolo12n.pt", help="Path to YOLO model")
    parser.add_argument("--model-type", type=str, choices=["yolo12"], default="yolo12", 
                        help="Type of YOLO model to use (yolo12)")
    parser.add_argument("--confidence", type=float, default=0.3, help="Detection confidence threshold")
    parser.add_argument("--output", type=str, default="", help="Path to output video file")
    parser.add_argument("--show", action="store_true", help="Display the video while processing")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--classes", type=int, nargs="+", default=[0], help="Classes to detect (default: 0 for person)")
    return parser.parse_args()

def process_video(video_path, model_path, confidence=0.3, classes=[0], 
                 output_path="people_detection_output.mp4", show=False, verbose=False):
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
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
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
        
        # Process each detection
        for i, (xyxy, _confidence, class_id) in enumerate(zip(
            detections.xyxy, detections.confidence, detections.class_id
        )):
            # Draw bounding box
            x1, y1, x2, y2 = xyxy
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            # Draw class and confidence
            label = f"{model.model.names[class_id]}: {_confidence:.2f}"
            cv2.putText(frame, label, (int(x1), int(y1) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw detection count
        cv2.putText(frame, f"Detections: {frame_detections}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Display the frame
        if show:
            cv2.imshow('People Detector', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Write the frame to output video
        output_writer.write(frame)
    
    # Close progress bar
    progress_bar.close()
    
    # Release resources
    cap.release()
    output_writer.release()
    cv2.destroyAllWindows()
    
    # Return results
    return output_path, frame_count, total_detections

def main():
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
        verbose=args.verbose
    )
    
    # Print results
    if output_path:
        print(f"Processing complete. {frame_count} frames processed.")
        print(f"Total detections: {detection_count}")

if __name__ == "__main__":
    main()
