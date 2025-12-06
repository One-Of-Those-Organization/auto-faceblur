# Auto-Faceblur

Automatically detect and blur faces in images and videos using AI with support for whitelist functionality.

## Overview

Auto-Faceblur is a computer vision tool that automatically detects and anonymizes faces in media files using modern AI models including YOLOv11 for accurate face detection. 

## Features

- Image and video processing with multiple format support
- High accuracy face detection using YOLOv11n model
- Customizable blur methods (Gaussian blur and pixelation)
- Adjustable confidence threshold for detection sensitivity
- Whitelist support to exclude specific faces from blurring
- CPU and GPU acceleration support when available
- Optimized performance for real-time processing

## Requirements

- Python 3.8+
- Dependencies listed in requirements.txt

## Installation

1.  Clone the repository
   ```bash
   git clone https://github.com/One-Of-Those-Organization/auto-faceblur.git
   cd auto-faceblur
   ```

2. Create a virtual environment (recommended)
   ```bash
   python -m venv . venv
   source .venv/bin/activate   # macOS / Linux
   .venv\Scripts\activate      # Windows PowerShell
   ```

3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure

```
auto-faceblur/
├── src/              # Source code directory
├── model/            # AI model files
├── weight/           # Model weights
├── requirements. txt # Python dependencies
└── README.md         # Documentation
├── OldReadme.md      # Old Documentation
```

## Technical Details

### AI Models
- **Face Detection**: YOLOv11n optimized for face detection
- **Framework**: OpenCV, TensorFlow/tf-keras, Ultralytics YOLO
- **Performance**: Optimized for both CPU and GPU processing

## Credits

- Face detection model: [YOLOv11n-face-detection by AdamCodd](https://huggingface.co/AdamCodd/YOLOv11n-face-detection/blob/main/model.pt)

## Performance Notes

The tool has been optimized for:
- Efficient face detection using YOLOv11
- Memory optimization for processing large files
- Multi-threading support for batch operations
- Real-time processing capabilities