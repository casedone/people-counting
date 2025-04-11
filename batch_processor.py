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
            # Read CSV file
            df = pd.read_csv(self.job_file)
            
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
    
    def process_job(self, job: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single job.
        
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
        
        # Process the video
        start_time = time.time()
        
        result_path, frame_count, up_count, down_count = people_counter.process_video(
            video_path=video_path,
            line_start=line_start,
            line_end=line_end,
            model_path=self.model_path,
            confidence=float(confidence),
            classes=classes,
            output_path=output_path,
            show=False  # Don't show the video while processing
        )
        
        processing_time = time.time() - start_time
        
        if result_path is None:
            raise RuntimeError(f"Error processing video: {video_file}")
        
        # Generate statistics
        stats = {
            'video_file': video_file,
            'output_file': output_path,
            'frame_count': frame_count,
            'up_count': up_count,
            'down_count': down_count,
            'total_count': up_count + down_count,
            'processing_time': processing_time,
            'fps': frame_count / processing_time if processing_time > 0 else 0,
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        # Save statistics to a text file
        stats_filename = os.path.splitext(output_filename)[0] + ".txt"
        stats_path = os.path.join(self.output_dir, stats_filename)
        
        with open(stats_path, 'w') as f:
            f.write(f"Video: {video_file}\n")
            f.write(f"Line: from {line_start} to {line_end}\n")
            f.write(f"Frames processed: {frame_count}\n")
            f.write(f"Processing time: {processing_time:.2f} seconds\n")
            f.write(f"FPS: {stats['fps']:.2f}\n")
            f.write(f"People count - Up: {up_count}, Down: {down_count}, Total: {up_count + down_count}\n")
            f.write(f"Output video: {output_path}\n")
            f.write(f"Timestamp: {stats['timestamp']}\n")
        
        # Return job results
        return {
            'job': job,
            'status': 'success',
            'stats': stats,
            'output_file': output_path,
            'stats_file': stats_path
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
                if result['status'] == 'success':
                    summary_data.append({
                        'video_file': result['job']['video_file'],
                        'status': result['status'],
                        'output_file': os.path.basename(result['output_file']),
                        'frame_count': result['stats']['frame_count'],
                        'up_count': result['stats']['up_count'],
                        'down_count': result['stats']['down_count'],
                        'total_count': result['stats']['total_count'],
                        'processing_time': f"{result['stats']['processing_time']:.2f}",
                        'fps': f"{result['stats']['fps']:.2f}"
                    })
                else:
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
        # Create batch processor
        processor = BatchProcessor(
            job_file=args.job_file,
            model_path=args.model,
            confidence=args.confidence,
            output_dir=args.output_dir
        )
        
        # Process all jobs
        print(f"Starting batch processing of {len(processor.jobs)} jobs...")
        results = processor.process_all()
        
        # Save summary
        summary_path = processor.save_summary(results, format=args.summary_format)
        print(f"\nBatch processing complete. Summary saved to: {summary_path}")
        
        # Count successes and failures
        successes = sum(1 for r in results if r['status'] == 'success')
        failures = sum(1 for r in results if r['status'] == 'failed')
        
        print(f"Processed {len(results)} jobs: {successes} successful, {failures} failed.")
        
        if successes > 0:
            # Calculate total counts
            total_up = sum(r['stats']['up_count'] for r in results if r['status'] == 'success')
            total_down = sum(r['stats']['down_count'] for r in results if r['status'] == 'success')
            total_count = total_up + total_down
            
            print(f"Total people counted - Up: {total_up}, Down: {total_down}, Total: {total_count}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
