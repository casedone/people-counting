#!/usr/bin/env python3
import os
import sys
import subprocess

def main():
    # Get the path to ffmpeg
    ffmpeg_path = "/usr/local/bin/ffmpeg"
    
    # Construct the ffprobe command using ffmpeg
    # ffmpeg can provide similar information with the right arguments
    args = [ffmpeg_path, "-i"] + sys.argv[1:] + ["-hide_banner"]
    
    # Run the command and capture the output
    try:
        result = subprocess.run(args, capture_output=True, text=True)
        # Print the output to stderr (where ffprobe normally outputs)
        sys.stderr.write(result.stderr)
        # Return the same exit code
        sys.exit(result.returncode)
    except Exception as e:
        sys.stderr.write(f"Error: {str(e)}\n")
        sys.exit(1)

if __name__ == "__main__":
    main()
