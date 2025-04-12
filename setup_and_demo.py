#!/usr/bin/env python3
import os
import sys
import subprocess
import argparse
import cv2  # Import OpenCV for video capture

def run_command(command):
    """Run a shell command and print output"""
    print(f"Running: {command}")
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    
    # Print output in real-time
    for line in iter(process.stdout.readline, b''):
        sys.stdout.write(line.decode('utf-8'))
    
    process.wait()
    return process.returncode

def download_file(url, output_path, redownload=False):
    """Download a file from a URL"""
    import requests
    
    if os.path.exists(output_path) and not redownload:
        print(f"File already exists: {output_path}")
        return True
    
    print(f"Downloading {url} to {output_path}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"Download complete: {output_path}")
        return True
    except Exception as e:
        print(f"Error downloading file: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Setup and demo for people counting application")
    parser.add_argument("--model-type", choices=["yolo12", "yolo11"], default="yolo12", 
                        help="Type of YOLO model to use (yolo12)")
    parser.add_argument("--model-size", choices=["n", "s", "m", "l", "x", "all"], default="all", 
                        help="YOLO model size: n(ano), s(mall), m(edium), l(arge), x(large), or all")
    parser.add_argument("--version", choices=["v8.3.0"], default="v8.3.0", 
                        help="Version of repo models to download. Default is v8.3.0. Check ultralytics website for more information.")
    parser.add_argument("--no-demo", action="store_true", help="Skip running the demo")
    parser.add_argument("--custom-video", type=str, help="Path to a custom video for the demo")
    parser.add_argument("--redownload", action="store_true", default=False, 
                        help="Force redownload of YOLO models even if they already exist")
    args = parser.parse_args()
    
    # Create necessary directories if they don't exist
    for directory in ["input", "output", "models"]:
        os.makedirs(directory, exist_ok=True)
        print(f"Ensured directory exists: {directory}")

    # Download YOLO models
    model_type = args.model_type
    model_size = args.model_size
    redownload = args.redownload  # Capture the redownload argument
    version = args.version
    
    # Define model sizes to download
    if model_size == "all":
        model_sizes = ["n", "s", "m", "l", "x"]
    else:
        model_sizes = [model_size]
    
    # Download each model
    for size in model_sizes:
        model_file = f"{model_type}{size}.pt"
        model_path = os.path.join("models", model_file)
        
        if not os.path.exists(model_path) or redownload:
            print(f"Downloading {model_type.upper()}{size} model...")
            model_url = f"https://github.com/ultralytics/assets/releases/download/{version}/{model_type}{size}.pt"
            
            if not download_file(model_url, model_path, redownload):
                print(f"Failed to download {model_file}. Please download it manually.")
                continue
        else:
            print(f"{model_type.upper()} model already exists: {model_path}")
    
    # Use the smallest model for demo if all were downloaded
    if model_size == "all":
        model_size = "n"
    
    model_file = os.path.join("models", f"{model_type}{model_size}.pt")
    
    if args.no_demo:
        print("Setup complete. Skipping demo.")
        return
    
    # Download sample video if needed
    video_path = args.custom_video
    if not video_path:
        sample_video = "sample_pedestrians.mp4"
        if not os.path.exists(sample_video):
            print("Downloading sample pedestrian video...")
            # This is a sample pedestrian video from Pexels (free to use)
            video_url = "https://www.pexels.com/download/video/5953790/"
            if not download_file(video_url, sample_video):
                print("Failed to download sample video. Please provide your own video using --custom-video.")
                return
        video_path = sample_video
    
    # Run the people counter
    print("\nRunning people counter demo...")
    
    try:
        # Try to use the process_video function directly
        import people_counter
        
        # Set up local ffmpeg and ffprobe binaries
        try:
            ffmpeg_path, ffprobe_path = people_counter.setup_local_ffmpeg()
            print("Successfully configured local ffmpeg and ffprobe binaries")
        except Exception as e:
            print(f"Warning: Failed to set up local ffmpeg/ffprobe: {e}")
            print("Falling back to system-installed ffmpeg/ffprobe if available")
        
        # Output directory already created at the beginning
        output_dir = os.path.join(os.getcwd(), "output")
        
        # Generate a unique filename based on timestamp
        import time
        timestamp = int(time.time())
        output_path = os.path.join(output_dir, f"people_counting_demo_{timestamp}.mp4")
        
        # Get video properties to set default line
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            # Default to a horizontal line in the middle of the frame
            line_start = [0, frame_height // 2]
            line_end = [frame_width, frame_height // 2]
            cap.release()
        else:
            # Fallback if can't open video
            line_start = [0, 0]
            line_end = [0, 0]
        
        print(f"Processing video: {video_path}")
        print(f"Line: from {line_start} to {line_end}")
        print(f"Model: {model_file}")
        print(f"Output: {output_path}")
        
        # Process the video
        result_path, frame_count, up_count, down_count = people_counter.process_video(
            video_path=video_path,
            line_start=line_start,
            line_end=line_end,
            model_path=model_file,
            confidence=0.3,
            classes=[0],  # Class 0 is person in COCO dataset
            output_path=output_path,
            show=True  # Show the video while processing
        )
        
        if result_path:
            print(f"\nProcessing complete. {frame_count} frames processed.")
            print(f"People count - Up: {up_count}, Down: {down_count}, Total: {up_count + down_count}")
            print(f"Output video saved to: {output_path}")
        else:
            print("Error processing video")
            
    except ImportError:
        # Fall back to running the command if import fails
        print("Could not import people_counter module. Using command line approach instead.")
        run_command(f"python people_counter.py --video {video_path} --model {model_file} --model-type {model_type} --show")
    
    print("\nDemo complete!")
    print("You can run the people counter on your own videos using:")
    print(f"python people_counter.py --video path/to/your/video.mp4 --model {model_file} --model-type {model_type} --show")
    print("\nOr use the line_selector.py tool to interactively select a counting line:")
    print(f"python line_selector.py --video path/to/your/video.mp4 --model-type {model_type}")
    print("\nOr use the gradio_interface.py for a web-based interface:")
    print("python gradio_interface.py")

if __name__ == "__main__":
    main()
