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

3. Download a YOLO model (if you don't have one already):
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

## License

This project is licensed under the MIT License - see the LICENSE file for details.
