# Eye & Mouth Masker

A real-time facial feature masking application that detects and blurs eyes and mouth using computer vision and machine learning.

## Overview

This application uses your webcam to detect facial landmarks (eyes and mouth) in real-time and applies customizable blurring to mask those regions. It features an interactive GUI built with Tkinter that allows you to adjust masking intensity and manually control feature regions using your mouse.



Uploading 20260511_BlitheHelplessClipsdadCclamChamp-Zi2SXP5v59AESlV6_source.mp4…



## Features

- **Real-time Face Detection**: Uses MediaPipe Face Landmarker for accurate facial landmark detection
- **Multiple Detection Methods**: Falls back to OpenCV Haar Cascades if MediaPipe is unavailable
- **Interactive UI**: Start/stop controls and real-time status updates
- **Adjustable Blur Strength**: Slider to control the feather/blur intensity (0-50)
- **Manual Region Control**: Click and drag to adjust the position and scale of masked regions
- **Smooth Tracking**: Exponential Moving Average (EMA) smoothing for stable detection across frames
- **Fullscreen Display**: Automatically maximizes to fullscreen for immersive preview
- **Performance Optimized**: Configurable detection intervals to balance accuracy and speed

## Requirements

- Python 3.8+
- Webcam access
- Windows, macOS, or Linux

## Installation

1. **Clone or download this repository**

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**:
   - **Windows**:
     ```bash
     venv\Scripts\activate
     ```
   - **macOS/Linux**:
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Start the application**:
   ```bash
   python app.py
   ```

2. **Using the Interface**:
   - Click **"Start Webcam"** to begin capturing and processing video
   - Use the **"Mask Feather"** slider to adjust blur intensity (higher = more blur)
   - Click **"Stop Webcam"** to stop capturing

3. **Manual Region Control**:
   - **Click and drag** on any detected region to move it
   - **Scroll/Mouse wheel** while hovering over a region to scale it (0.35x to 3.0x)
   - Regions will revert to auto-detection after a brief period of inactivity

## How It Works

### Detection Methods

1. **Primary**: MediaPipe Face Landmarker
   - High-accuracy neural network-based detection
   - Detects precise facial landmarks for accurate region identification
   - Model automatically downloads on first run (~200MB)

2. **Fallback**: OpenCV Haar Cascades
   - Used if MediaPipe is unavailable
   - Includes cascades for face, eyes, and mouth detection
   - Faster but less accurate

### Processing Pipeline

1. **Frame Capture**: Reads frames from webcam at preferred resolution (1280x720 or fallback to 640x480)
2. **Face Detection**: Detects facial landmarks or face regions
3. **Feature Region Tracking**: Identifies eyes and mouth, applies EMA smoothing
4. **Masking**: Applies Gaussian blur to detected regions based on slider value
5. **Display**: Renders masked frame in real-time UI

### Key Parameters

- **`blur_strength`**: Feathering/blur radius for masked regions (0-50)
- **`detection_interval`**: Run detection every N frames (1 = every frame)
- **`landmark_smooth_alpha`**: EMA weight for smooth tracking (0.65 default)
- **`detection_max_dim`**: Maximum dimension for detection (480px for speed)

## File Structure

```
augegit/
├── app.py                      # Main application code
├── requirements.txt            # Python dependencies
├── models/
│   └── face_landmarker.task   # MediaPipe Face Landmarker model
└── README.md                   # This file
```

## Dependencies

- **opencv-python**: Computer vision and image processing
- **pillow**: Image handling for Tkinter display
- **mediapipe**: Pre-trained ML models for face landmark detection

## Troubleshooting

### Camera won't open
- Ensure your camera is connected and not in use by another application
- Try unplugging and reconnecting your camera
- Check camera permissions in your system settings

### MediaPipe model download fails
- Check your internet connection
- The model (~200MB) will download to `.models/face_landmarker.task` on first run
- If download times out, try running again

### Blurring is slow/laggy
- Reduce blur strength slightly
- Try enabling detection skipping (modify `detection_interval` to 2 or 3)
- Lower camera resolution settings

### Face not detected
- Ensure adequate lighting
- Face should be clearly visible to camera
- Try adjusting camera angle and distance
- The app will fall back to Haar Cascades if Face Landmarker has issues

## Performance Notes

- Target: 30 FPS video capture
- Detection runs by default every frame (adjustable via `detection_interval`)
- Memory usage: ~300-500MB depending on model and resolution
- GPU acceleration available through MediaPipe but not required

## License

This project uses MediaPipe (Apache 2.0) and OpenCV (Apache 2.0).

## Future Enhancements

- Additional facial regions (nose, face outline)
- Pixelation as alternative to blur
- Frame recording/export
- Batch processing for video files
- Settings persistence
