# People Counting with YOLOv8, YOLOv10, and YOLOv11

This application counts people crossing a defined line in a video using YOLOv8, YOLOv10, or YOLOv11 for detection and tracking.

## Features

- People detection using YOLOv8, YOLOv10, or YOLOv11
- Two counting approaches:
  - Traditional: Custom LineCounter with ByteTrack for object tracking
  - Solutions: Ultralytics solutions.ObjectCounter for integrated object counting
- Line crossing detection with directional counting (up/down)
- Visualization of detections, tracking IDs, and count
- Option to save processed video with annotations
- Gradio web interface for easy interaction
- Batch processing for multiple videos

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/people-counting.git
cd people-counting
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. FFmpeg Requirements:
   - This project requires FFmpeg for video processing
   - The Gradio interface specifically needs both ffmpeg and ffprobe
   - **Important**: Place the ffmpeg and ffprobe binary files in the `bin/` folder during setup
     - The application is configured to use these local binaries instead of system-installed versions
     - This ensures consistent behavior across different environments

4. Download a YOLO model (if you don't have one already):
```bash
# YOLOv8 models
# For a small model (fastest)
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt

# For a medium model (balanced)
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt

# For a large model (most accurate)
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt

# YOLOv10 models
# For a small model (fastest)
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov10n.pt

# For a medium model (balanced)
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov10m.pt

# For a large model (most accurate)
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov10l.pt
```

## Quick Start

The easiest way to get started is to use the setup and demo script:

```bash
python setup_and_demo.py
```

This script will:
1. Install all required dependencies
2. Download a YOLOv8 model (nano by default)
3. Download a sample pedestrian video
4. Run the people counter on the sample video

You can customize the setup with these options:
- `--model-type`: Choose model type (yolov8, yolov10, or yolov11) - default is 'yolov8'
- `--model-size`: Choose model size (n, s, m, l, x) - default is 'n' (nano)
- `--use-solutions`: Use Ultralytics solutions.ObjectCounter instead of custom LineCounter
- `--no-demo`: Skip running the demo
- `--custom-video`: Use your own video instead of downloading the sample

Examples:
```bash
# Use YOLOv8 medium model
python setup_and_demo.py --model-type yolov8 --model-size m

# Use YOLOv10 nano model
python setup_and_demo.py --model-type yolov10 --model-size n

# Use YOLOv11 nano model with solutions.ObjectCounter
python setup_and_demo.py --model-type yolov11 --model-size n --use-solutions
```

## Usage

Run the people counter on a video file:

```bash
python people_counter.py --video path/to/your/video.mp4 --show
```

### Command Line Arguments

- `--video`: Path to the input video file (required)
- `--model`: Path to the YOLO model file (default: "yolov8n.pt")
- `--model-type`: Type of YOLO model to use (choices: "yolov8", "yolov10", "yolov11", default: "yolov8")
- `--line-start`: Starting point of the counting line as "x y" (default: middle of the frame)
- `--line-end`: Ending point of the counting line as "x y" (default: middle of the frame)
- `--confidence`: Detection confidence threshold (default: 0.3)
- `--output`: Path to save the output video (optional)
- `--show`: Display the video while processing (optional)
- `--use-solutions`: Use Ultralytics solutions.ObjectCounter instead of custom LineCounter
- `--classes`: Classes to detect (default: 0 for person) - can specify multiple classes

### Interactive Line Selection

To interactively select the counting line, use the line_selector.py utility:

```bash
python line_selector.py --video path/to/your/video.mp4
```

This tool will:
1. Allow you to select a frame from the video using a trackbar
2. Let you draw a counting line by clicking and dragging on the frame
3. Generate the command to run the people counter with your custom line
4. Optionally run the command for you

### Examples

Count people crossing a horizontal line in the middle of the frame:
```bash
python people_counter.py --video pedestrians.mp4 --show
```

Count people crossing a custom line and save the output:
```bash
python people_counter.py --video pedestrians.mp4 --line-start 100 300 --line-end 500 300 --output result.mp4 --show
```

Use different YOLO models:
```bash
# Use YOLOv8 medium model
python people_counter.py --video pedestrians.mp4 --model yolov8m.pt --model-type yolov8 --show

# Use YOLOv10 nano model
python people_counter.py --video pedestrians.mp4 --model yolov10n.pt --model-type yolov10 --show

# Use YOLOv11 nano model with solutions.ObjectCounter
python people_counter.py --video pedestrians.mp4 --model yolov11n.pt --model-type yolov11 --use-solutions --show

# Count multiple object classes (person and car)
python people_counter.py --video traffic.mp4 --classes 0 2 --use-solutions --show
```

## Folder Cleaner

This module allows you to clear the input, output, and batch_jobs folders individually or in any combination.

### Usage

```bash
python folder_cleaner.py [options]
```

### Command Line Arguments

- `--input`: Clear the input folder
- `--output`: Clear the output folder
- `--batch-jobs`: Clear the batch_jobs folder
- `--all`: Clear all folders (input, output, and batch_jobs)
- `--delete-gitkeep`: Also delete .gitkeep files (by default, .gitkeep files are preserved)

### Examples

Clear the input folder:
```bash
python folder_cleaner.py --input
```

Clear both output and batch_jobs folders:
```bash
python folder_cleaner.py --output --batch-jobs
```

Clear all folders:
```bash
python folder_cleaner.py --all
```

Clear all folders including .gitkeep files:
```bash
python folder_cleaner.py --all --delete-gitkeep
```

### Using in Python Code

You can also use the folder cleaner functions in your Python code:

```python
import folder_cleaner

# Clear individual folders
folder_cleaner.clear_input_folder()
folder_cleaner.clear_output_folder()
folder_cleaner.clear_batch_jobs_folder()

# Clear multiple folders
folder_cleaner.clear_folders(['input', 'output'])

# Clear all folders
folder_cleaner.clear_all_folders()
```

## Batch Processing

This module allows you to process multiple videos for people counting in batch mode. It takes a job file that specifies video filenames and counting line coordinates, processes each video, and saves the results in the output folder.

### Usage

```bash
python batch_processor.py --job-file path/to/job_file.json
```

### Command Line Arguments

- `--job-file`: Path to the job file (JSON or CSV format) [required]
- `--model`: Path to the YOLO model (default: "models/yolo12n.pt")
- `--confidence`: Detection confidence threshold (default: 0.3)
- `--output-dir`: Directory to save output videos and statistics (default: "output")
- `--summary-format`: Format for the summary file (csv or json, default: csv)

### Job File Format

The job file can be in either JSON or CSV format.

#### JSON Format

```json
{
  "jobs": [
    {
      "video_file": "input/video1.mp4",
      "line_start": [0, 360],
      "line_end": [640, 360],
      "confidence": 0.3
    },
    {
      "video_file": "input/video2.mp4",
      "line_start": [320, 0],
      "line_end": [320, 720],
      "confidence": 0.4
    }
  ]
}
```

Alternatively, you can use a simpler format with just an array of jobs:

```json
[
  {
    "video_file": "input/video1.mp4",
    "line_start": [0, 360],
    "line_end": [640, 360],
    "confidence": 0.3
  },
  {
    "video_file": "input/video2.mp4",
    "line_start": [320, 0],
    "line_end": [320, 720],
    "confidence": 0.4
  }
]
```

#### CSV Format

```csv
video_file,line_start,line_end,confidence
input/video1.mp4,"0,360","640,360",0.3
input/video2.mp4,"320,0","320,720",0.4
```

### Job Parameters

- `video_file`: Path to the video file (relative to the working directory) [required]
- `line_start`: Starting point of the counting line (x, y) [required]
- `line_end`: Ending point of the counting line (x, y) [required]
- `confidence`: Detection confidence threshold (optional, defaults to the value provided in command line)
- `classes`: List of classes to detect (optional, defaults to [0] for person)

### Output

For each processed video, the following outputs are generated:

1. **Processed Video**: A video file with bounding boxes and counting visualization
2. **Statistics Text File**: A text file with counting statistics
3. **Summary File**: A CSV or JSON file summarizing all processed jobs

#### Example Summary CSV

```csv
video_file,status,output_file,frame_count,up_count,down_count,total_count,processing_time,fps
input/video1.mp4,success,video1_counting_20250411_185432.mp4,1200,15,12,27,45.32,26.48
input/video2.mp4,success,video2_counting_20250411_185517.mp4,900,8,10,18,35.67,25.23
```

### Examples

Process videos using a JSON job file:

```bash
python batch_processor.py --job-file sample_jobs.json
```

Process videos using a CSV job file with a custom model:

```bash
python batch_processor.py --job-file sample_jobs.csv --model models/yolo12m.pt
```

Process videos with a higher confidence threshold:

```bash
python batch_processor.py --job-file sample_jobs.json --confidence 0.5
```

Save the summary in JSON format:

```bash
python batch_processor.py --job-file sample_jobs.json --summary-format json
```

### Tips

1. Make sure all video files exist in the specified paths.
2. The line coordinates should be within the video frame dimensions.
3. For horizontal counting lines, set different x-coordinates and the same y-coordinate.
4. For vertical counting lines, set different y-coordinates and the same x-coordinate.
5. The "up" and "down" counts are relative to the line direction. For a horizontal line, "up" means crossing from bottom to top, and "down" means crossing from top to bottom.

## How It Works

### Line Crossing Detection

The application defines a counting line and a region around it. When a person's center point crosses this line, they are counted. The direction of crossing (up or down) is determined by the sign change of the distance from the line.

- **Green bounding box**: Person detected but not crossing the line
- **Red bounding box**: Person crossing the line in the upward direction
- **Blue bounding box**: Person crossing the line in the downward direction
- **Purple line**: The counting line
- **Purple region**: The counting region around the line

### Tracking

The application uses ByteTrack to maintain consistent tracking IDs for each person. This ensures that each person is counted only once when they cross the line.

## Troubleshooting

- **Model not found**: Make sure you've downloaded the YOLOv8 model and specified the correct path.
- **Video not found**: Check that the video file exists and the path is correct.
- **Low detection accuracy**: Try using a larger YOLOv8 model (yolov8m.pt or yolov8l.pt) or adjust the confidence threshold.
- **Missed crossings**: Adjust the counting region size (modify the `counting_region` parameter in the code).
- **Memory issues**: If you encounter memory problems with large videos, try using a smaller YOLOv8 model.
- **Video playback issues in Gradio interface**: 
  - Ensure the ffmpeg and ffprobe binaries are correctly placed in the `bin/` folder
  - Check that the binaries have executable permissions (see the next troubleshooting item)

- **Issues with ffmpeg/ffprobe binaries**:
  - Make sure the ffmpeg and ffprobe binaries are placed in the `bin/` folder
  - Ensure the binaries have executable permissions:
    ```bash
    chmod +x bin/ffmpeg bin/ffprobe
    ```
  - If you're still experiencing issues, you can check if the application is correctly detecting the binaries by running:
    ```bash
    python -c "from people_counter import setup_local_ffmpeg; setup_local_ffmpeg()"
    ```
  - This should output the paths to the detected ffmpeg and ffprobe binaries

## License

This project is licensed under the MIT License - see the LICENSE file for details.
