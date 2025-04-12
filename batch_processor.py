#!/usr/bin/env python3
import os
import sys
import json
import csv
import argparse
import datetime
import time
from pathlib import Path
import pandas as pd
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

# Import the people counting functionality
import people_counter

class BatchProcessor:
    """
    Batch processor for people counting videos.
    Processes multiple videos based on a job file.
    """
    
    def __init__(self, job_file: str, model_path: str = "models/yolo12n.pt", 
                 confidence: float = 0.3, output_dir: str = "output"):
        """
        Initialize the batch processor.
        
        Args:
            job_file: Path to the job file
            model_path: Path to the YOLO model
            confidence: Detection confidence threshold
            output_dir: Directory to save output videos and statistics
        """
        self.job_file = job_file
        self.model_path = model_path
        self.confidence = confidence
        self.output_dir = output_dir
        self.jobs = []
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Load jobs from the job file
        self._load_jobs()
    
    def _load_jobs(self):
        """Load jobs from the job file."""
        if not os.path.exists(self.job_file):
            raise FileNotFoundError(f"Job file not found: {self.job_file}")
        
        file_ext = os.path.splitext(self.job_file)[1].lower()
        
        if file_ext == '.json':
            self._load_jobs_from_json()
        elif file_ext == '.csv':
            self._load_jobs_from_csv()
        else:
            raise ValueError(f"Unsupported job file format: {file_ext}. Use .json or .csv")
    
    def _load_jobs_from_json(self):
        """Load jobs from a JSON file."""
        try:
            with open(self.job_file, 'r') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                self.jobs = data
            elif isinstance(data, dict) and 'jobs' in data:
                self.jobs = data['jobs']
            else:
                raise ValueError("Invalid JSON format. Expected a list of jobs or a dict with a 'jobs' key.")
            
            # Validate jobs
            self._validate_jobs()
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON file: {str(e)}")
    
    def _load_jobs_from_csv(self):
        """Load jobs from a CSV file."""
        try:
            # Read CSV file with more flexible parsing options
            # skipinitialspace=True handles spaces after commas
            df = pd.read_csv(self.job_file, skipinitialspace=True)
            
            # Convert DataFrame to list of dictionaries
            self.jobs = df.to_dict(orient='records')
            
            # Validate jobs
            self._validate_jobs()
            
        except Exception as e:
            raise ValueError(f"Error reading CSV file: {str(e)}")
    
    def _validate_jobs(self):
        """Validate job entries."""
        valid_jobs = []
        
        for i, job in enumerate(self.jobs):
            # Check required fields
            if 'video_file' not in job:
                print(f"Warning: Job {i+1} missing 'video_file' field, skipping.")
                continue
            
            if 'line_start' not in job or 'line_end' not in job:
                print(f"Warning: Job {i+1} missing line coordinates, skipping.")
                continue
            
            # Validate line coordinates
            try:
                # Handle different formats of line coordinates
                if isinstance(job['line_start'], list) and len(job['line_start']) == 2:
                    line_start = [int(job['line_start'][0]), int(job['line_start'][1])]
                elif isinstance(job['line_start'], str):
                    # Try to parse string format like "100,200"
                    coords = job['line_start'].split(',')
                    if len(coords) == 2:
                        line_start = [int(coords[0]), int(coords[1])]
                    else:
                        raise ValueError("Invalid line_start format")
                else:
                    raise ValueError("Invalid line_start format")
                
                if isinstance(job['line_end'], list) and len(job['line_end']) == 2:
                    line_end = [int(job['line_end'][0]), int(job['line_end'][1])]
                elif isinstance(job['line_end'], str):
                    # Try to parse string format like "100,200"
                    coords = job['line_end'].split(',')
                    if len(coords) == 2:
                        line_end = [int(coords[0]), int(coords[1])]
                    else:
                        raise ValueError("Invalid line_end format")
                else:
                    raise ValueError("Invalid line_end format")
                
                # Update job with parsed coordinates
                job['line_start'] = line_start
                job['line_end'] = line_end
                
                # Add to valid jobs
                valid_jobs.append(job)
                
            except (ValueError, TypeError) as e:
                print(f"Warning: Job {i+1} has invalid line coordinates: {str(e)}, skipping.")
                continue
        
        self.jobs = valid_jobs
        print(f"Loaded {len(self.jobs)} valid jobs.")
    
    def test_jobs(self) -> Tuple[List[Dict[str, Any]], bool]:
        """
        Test all jobs to check if they can be loaded and processed without actually processing them.
        
        Returns:
            Tuple containing:
            - List of test results for each job
            - Boolean indicating if any errors were found
        """
        test_results = []
        has_errors = False
        
        print(f"Testing {len(self.jobs)} jobs...")
        
        for i, job in enumerate(self.jobs):
            print(f"Testing job {i+1}/{len(self.jobs)}: {job['video_file']}")
            
            try:
                # Extract job parameters
                video_file = job['video_file']
                line_start = job['line_start']
                line_end = job['line_end']
                
                # Get optional parameters with defaults
                confidence = job.get('confidence', self.confidence)
                classes = job.get('classes', [0])  # Default to class 0 (person)
                
                # Check if video file exists
                video_path = os.path.join(os.getcwd(), video_file)
                if not os.path.isfile(video_path):
                    raise FileNotFoundError(f"Video file not found: {video_path}")
                
                # Check if model exists
                if not os.path.isfile(self.model_path):
                    raise FileNotFoundError(f"Model file not found: {self.model_path}")
                
                # Try to open the video to check if it's valid
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    raise RuntimeError(f"Could not open video: {video_path}")
                
                # Read the first frame to check if video is readable
                ret, _ = cap.read()
                if not ret:
                    raise RuntimeError(f"Could not read frames from video: {video_path}")
                
                # Release the video capture
                cap.release()
                
                # Add successful test to results
                test_results.append({
                    'job': job,
                    'status': 'success'
                })
                print(f"Job {i+1} test passed.")
                
            except Exception as e:
                has_errors = True
                print(f"Error testing job {i+1}: {str(e)}")
                # Add failed test to results
                test_results.append({
                    'job': job,
                    'status': 'failed',
                    'error': str(e)
                })
        
        # Print summary
        success_count = sum(1 for r in test_results if r['status'] == 'success')
        failed_count = sum(1 for r in test_results if r['status'] == 'failed')
        
        print(f"\nTest summary: {success_count} jobs passed, {failed_count} jobs failed.")
        
        return test_results, has_errors
    
    def process_all(self) -> List[Dict[str, Any]]:
        """
        Process all jobs in the job file.
        
        Returns:
            List of results for each job
        """
        results = []
        
        for i, job in enumerate(self.jobs):
            print(f"\nProcessing job {i+1}/{len(self.jobs)}: {job['video_file']}")
            
            try:
                result = self.process_job(job)
                results.append(result)
                print(f"Job {i+1} completed successfully.")
            except Exception as e:
                print(f"Error processing job {i+1}: {str(e)}")
                # Add failed job to results
                results.append({
                    'job': job,
                    'status': 'failed',
                    'error': str(e)
                })
        
        return results
    
    def _try_repair_frame(self, cap, frame_pos, max_attempts=3):
        """
        Try to repair a corrupted frame using multiple strategies.
        
        Args:
            cap: OpenCV VideoCapture object
            frame_pos: Position of the frame to repair
            max_attempts: Maximum number of repair attempts
            
        Returns:
            tuple: (success, frame)
        """
        # Strategy 1: Try to seek to the frame again
        for attempt in range(max_attempts):
            # Set position to the problematic frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                return True, frame
        
        # Strategy 2: Try to seek to the previous frame and then read next
        if frame_pos > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos - 1)
            # Read and discard the previous frame
            ret, _ = cap.read()
            if ret:
                # Now read the target frame
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    return True, frame
        
        # Strategy 3: Try to seek forward and then backward
        forward_pos = min(frame_pos + 5, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, forward_pos)
        ret, _ = cap.read()
        if ret:
            # Now seek back to our target frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                return True, frame
        
        # All repair strategies failed
        return False, None
    
    def process_job(self, job: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single job with robust frame-by-frame handling.
        
        Args:
            job: Job dictionary with video_file, line_start, and line_end
            
        Returns:
            Dictionary with job results
        """
        # Extract job parameters
        video_file = job['video_file']
        line_start = job['line_start']
        line_end = job['line_end']
        
        # Get optional parameters with defaults
        confidence = job.get('confidence', self.confidence)
        classes = job.get('classes', [0])  # Default to class 0 (person)
        
        # Check if video file exists
        video_path = os.path.join(os.getcwd(), video_file)
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Generate output filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        video_basename = os.path.splitext(os.path.basename(video_file))[0]
        output_filename = f"{video_basename}_counting_{timestamp}.mp4"
        output_path = os.path.join(self.output_dir, output_filename)
        
        # Generate corruption report filename
        corruption_report_filename = os.path.splitext(output_filename)[0] + "_corruptions.txt"
        corruption_report_path = os.path.join(self.output_dir, corruption_report_filename)
        
        # Open the video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        # Initialize YOLO model
        print(f"model_path = {self.model_path}")
        from ultralytics import YOLO
        import supervision as sv
        model = YOLO(self.model_path, task='detect', verbose=False)
        
        # Initialize tracker
        tracker = sv.ByteTrack()
        
        # Initialize line counter
        from people_counter import LineCounter
        line_counter = LineCounter(line_start, line_end)
        
        # Process the video frame by frame
        start_time = time.time()
        frame_count = 0
        corrupted_frames = []
        skipped_frames = []
        
        # Use tqdm for progress tracking
        from tqdm import tqdm
        progress_bar = tqdm(total=total_frames, desc="Processing video", 
                           unit="frames", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")
        
        current_frame_pos = 0
        while current_frame_pos < total_frames:
            # Try to read the frame
            ret, frame = cap.read()
            
            # Check if frame was read successfully
            if not ret or frame is None or frame.size == 0:
                # Frame is corrupted, try to repair it
                print(f"\nCorrupted frame detected at position {current_frame_pos} (timestamp: {current_frame_pos/fps:.2f}s)")
                corrupted_frames.append(current_frame_pos)
                
                # Try to repair the frame
                repair_success, repaired_frame = self._try_repair_frame(cap, current_frame_pos)
                
                if repair_success:
                    print(f"Successfully repaired frame at position {current_frame_pos}")
                    frame = repaired_frame
                else:
                    # Could not repair, skip this frame
                    print(f"Could not repair frame at position {current_frame_pos}, skipping")
                    skipped_frames.append(current_frame_pos)
                    current_frame_pos += 1
                    progress_bar.update(1)
                    continue
            
            # Process the frame with YOLO
            try:
                # Run YOLO inference on the frame
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
                
                # Write the frame to output video
                output_writer.write(frame)
                frame_count += 1
                
            except Exception as e:
                print(f"\nError processing frame {current_frame_pos}: {str(e)}")
                skipped_frames.append(current_frame_pos)
            
            # Move to next frame
            current_frame_pos += 1
            progress_bar.update(1)
        
        # Close progress bar
        progress_bar.close()
        
        # Get counts
        up_count = line_counter.up_count
        down_count = line_counter.down_count
        
        # Release resources
        cap.release()
        output_writer.release()
        cv2.destroyAllWindows()
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Write corruption report
        with open(corruption_report_path, 'w') as f:
            f.write(f"Video: {video_file}\n")
            f.write(f"Total frames: {total_frames}\n")
            f.write(f"Frames processed: {frame_count}\n")
            f.write(f"Corrupted frames detected: {len(corrupted_frames)}\n")
            f.write(f"Frames skipped: {len(skipped_frames)}\n\n")
            
            if corrupted_frames:
                f.write("Corrupted frames (position, timestamp):\n")
                for pos in corrupted_frames:
                    f.write(f"  Frame {pos}: {pos/fps:.2f}s\n")
                f.write("\n")
            
            if skipped_frames:
                f.write("Skipped frames (position, timestamp):\n")
                for pos in skipped_frames:
                    f.write(f"  Frame {pos}: {pos/fps:.2f}s\n")
        
        # Print summary
        print(f"\nVideo processing summary:")
        print(f"  Total frames in video: {total_frames}")
        print(f"  Frames processed: {frame_count}")
        print(f"  Corrupted frames detected: {len(corrupted_frames)}")
        print(f"  Frames skipped: {len(skipped_frames)}")
        print(f"  People count - Up: {up_count}, Down: {down_count}, Total: {up_count + down_count}")
        print(f"  Corruption report saved to: {corruption_report_path}")
        
        # Generate statistics
        stats = {
            'video_file': video_file,
            'output_file': output_path,
            'frame_count': frame_count,
            'total_frames': total_frames,
            'corrupted_frames': len(corrupted_frames),
            'skipped_frames': len(skipped_frames),
            'up_count': up_count,
            'down_count': down_count,
            'total_count': up_count + down_count,
            'processing_time': processing_time,
            'fps': frame_count / processing_time if processing_time > 0 else 0,
            'timestamp': datetime.datetime.now().isoformat(),
            'corruption_report': corruption_report_path
        }
        
        # Save statistics to a text file
        stats_filename = os.path.splitext(output_filename)[0] + ".txt"
        stats_path = os.path.join(self.output_dir, stats_filename)
        
        with open(stats_path, 'w') as f:
            f.write(f"Video: {video_file}\n")
            f.write(f"Line: from {line_start} to {line_end}\n")
            f.write(f"Total frames in video: {total_frames}\n")
            f.write(f"Frames processed: {frame_count}\n")
            f.write(f"Corrupted frames detected: {len(corrupted_frames)}\n")
            f.write(f"Frames skipped: {len(skipped_frames)}\n")
            f.write(f"Processing time: {processing_time:.2f} seconds\n")
            f.write(f"FPS: {stats['fps']:.2f}\n")
            f.write(f"People count - Up: {up_count}, Down: {down_count}, Total: {up_count + down_count}\n")
            f.write(f"Output video: {output_path}\n")
            f.write(f"Corruption report: {corruption_report_path}\n")
            f.write(f"Timestamp: {stats['timestamp']}\n")
        
        # Determine job status
        status = 'success'
        if len(skipped_frames) > 0:
            if frame_count / total_frames < 0.95:  # If less than 95% of frames were processed
                status = 'partial'
            else:
                status = 'success_with_skips'
        
        # Return job results
        return {
            'job': job,
            'status': status,
            'stats': stats,
            'output_file': output_path,
            'stats_file': stats_path,
            'corruption_report': corruption_report_path
        }
    
    def save_summary(self, results: List[Dict[str, Any]], format: str = 'csv') -> str:
        """
        Save a summary of all job results.
        
        Args:
            results: List of job results
            format: Output format ('csv' or 'json')
            
        Returns:
            Path to the summary file
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format.lower() == 'csv':
            # Create a list of dictionaries for the CSV
            summary_data = []
            
            for result in results:
                if result['status'] in ['success', 'success_with_skips', 'partial']:
                    # For success, success_with_skips, and partial results, include stats
                    entry = {
                        'video_file': result['job']['video_file'],
                        'status': result['status'],
                        'output_file': os.path.basename(result['output_file']),
                        'frame_count': result['stats']['frame_count'],
                        'total_frames': result['stats']['total_frames'],
                        'processed_percent': f"{(result['stats']['frame_count'] / result['stats']['total_frames'] * 100):.2f}%",
                        'corrupted_frames': result['stats']['corrupted_frames'],
                        'skipped_frames': result['stats']['skipped_frames'],
                        'up_count': result['stats']['up_count'],
                        'down_count': result['stats']['down_count'],
                        'total_count': result['stats']['total_count'],
                        'processing_time': f"{result['stats']['processing_time']:.2f}",
                        'fps': f"{result['stats']['fps']:.2f}"
                    }
                    
                    # Add error information for partial results
                    if result['status'] == 'partial':
                        entry['error'] = result.get('error', 'Unknown error')
                        
                    summary_data.append(entry)
                else:
                    # For failed results
                    summary_data.append({
                        'video_file': result['job']['video_file'],
                        'status': result['status'],
                        'error': result.get('error', 'Unknown error')
                    })
            
            # Save to CSV
            summary_path = os.path.join(self.output_dir, f"batch_summary_{timestamp}.csv")
            df = pd.DataFrame(summary_data)
            df.to_csv(summary_path, index=False)
            
        elif format.lower() == 'json':
            # Save to JSON
            summary_path = os.path.join(self.output_dir, f"batch_summary_{timestamp}.json")
            with open(summary_path, 'w') as f:
                json.dump(results, f, indent=2)
        
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'csv' or 'json'.")
        
        return summary_path


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Batch process videos for people counting")
    parser.add_argument("--job-file", type=str, required=True, 
                        help="Path to job file (JSON or CSV)")
    parser.add_argument("--model", type=str, default="models/yolo12n.pt", 
                        help="Path to YOLO model")
    parser.add_argument("--confidence", type=float, default=0.3, 
                        help="Detection confidence threshold")
    parser.add_argument("--output-dir", type=str, default="output", 
                        help="Directory to save output videos and statistics")
    parser.add_argument("--summary-format", type=str, choices=['csv', 'json'], default='csv',
                        help="Format for the summary file (csv or json)")
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_arguments()
    
    try:
        # Set up local ffmpeg and ffprobe binaries
        try:
            from people_counter import setup_local_ffmpeg
            ffmpeg_path, ffprobe_path = setup_local_ffmpeg()
            print("Successfully configured local ffmpeg and ffprobe binaries")
        except Exception as e:
            print(f"Warning: Failed to set up local ffmpeg/ffprobe: {e}")
            print("Falling back to system-installed ffmpeg/ffprobe if available")
        
        # Create batch processor
        processor = BatchProcessor(
            job_file=args.job_file,
            model_path=args.model,
            confidence=args.confidence,
            output_dir=args.output_dir
        )
        
        # First, test all jobs
        print(f"Testing {len(processor.jobs)} jobs before processing...")
        test_results, has_errors = processor.test_jobs()
        
        # If there are errors, ask user if they want to proceed
        proceed = True
        if has_errors:
            while True:
                user_input = input("\nSome jobs have errors. Do you want to proceed with processing? (y/n): ").lower()
                if user_input in ['y', 'yes']:
                    proceed = True
                    break
                elif user_input in ['n', 'no']:
                    proceed = False
                    break
                else:
                    print("Please enter 'y' or 'n'.")
        
        if proceed:
            # Process all jobs
            print(f"\nStarting batch processing of {len(processor.jobs)} jobs...")
            results = processor.process_all()
            
            # Save summary
            summary_path = processor.save_summary(results, format=args.summary_format)
            print(f"\nBatch processing complete. Summary saved to: {summary_path}")
            
            # Count successes, success_with_skips, partials, and failures
            successes = sum(1 for r in results if r['status'] == 'success')
            success_with_skips = sum(1 for r in results if r['status'] == 'success_with_skips')
            partials = sum(1 for r in results if r['status'] == 'partial')
            failures = sum(1 for r in results if r['status'] == 'failed')
            
            print(f"Processed {len(results)} jobs: {successes} successful, {success_with_skips} successful with skips, {partials} partial, {failures} failed.")
            
            if successes > 0 or success_with_skips > 0 or partials > 0:
                # Calculate total counts (include successful, success_with_skips, and partial results)
                total_up = sum(r['stats']['up_count'] for r in results if r['status'] in ['success', 'success_with_skips', 'partial'])
                total_down = sum(r['stats']['down_count'] for r in results if r['status'] in ['success', 'success_with_skips', 'partial'])
                total_count = total_up + total_down
                
                print(f"Total people counted - Up: {total_up}, Down: {total_down}, Total: {total_count}")
                
                if success_with_skips > 0:
                    print("\nNote: Some videos were processed successfully but had skipped frames.")
                    print("The counts from these videos should be accurate but may miss events in skipped frames.")
                
                if partials > 0:
                    print("\nNote: Some videos were only partially processed due to errors.")
                    print("The counts from these videos may be incomplete.")
        else:
            print("\nBatch processing cancelled by user.")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
