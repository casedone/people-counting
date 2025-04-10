#!/usr/bin/env python3
import os
import sys
import subprocess
import argparse

def run_command(command):
    """Run a shell command and print output"""
    print(f"Running: {command}")
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    
    # Print output in real-time
    for line in iter(process.stdout.readline, b''):
        sys.stdout.write(line.decode('utf-8'))
    
    process.wait()
    return process.returncode

def download_file(url, output_path):
    """Download a file from a URL"""
    import requests
    
    if os.path.exists(output_path):
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
    parser.add_argument("--model-size", choices=["n", "s", "m", "l", "x"], default="n", 
                        help="YOLOv8 model size: n(ano), s(mall), m(edium), l(arge), x(large)")
    parser.add_argument("--no-demo", action="store_true", help="Skip running the demo")
    parser.add_argument("--custom-video", type=str, help="Path to a custom video for the demo")
    args = parser.parse_args()
    
    # Install dependencies if needed
    if not os.path.exists("requirements.txt"):
        print("Error: requirements.txt not found. Please run this script from the project directory.")
        return
    
    print("Installing dependencies...")
    run_command("pip install -r requirements.txt")
    
    # Install additional dependencies for this script
    run_command("pip install requests")
    
    # Download YOLOv8 model
    model_size = args.model_size
    model_file = f"yolov8{model_size}.pt"
    
    if not os.path.exists(model_file):
        print(f"Downloading YOLOv8{model_size} model...")
        model_url = f"https://github.com/ultralytics/assets/releases/download/v0.0.0/{model_file}"
        if not download_file(model_url, model_file):
            print("Failed to download model. Please download it manually.")
            return
    else:
        print(f"YOLOv8 model already exists: {model_file}")
    
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
    run_command(f"python people_counter.py --video {video_path} --model {model_file} --show")
    
    print("\nDemo complete!")
    print("You can run the people counter on your own videos using:")
    print(f"python people_counter.py --video path/to/your/video.mp4 --model {model_file} --show")

if __name__ == "__main__":
    main()
