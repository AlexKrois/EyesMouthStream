# Eye & Mouth Masker

A real-time webcam application that intelligently detects and masks your eyes and mouth with adjustable blur and feathering effects. Available as both a **desktop application** (Python/tkinter) and a **web application** (HTML/JavaScript).

## 🎯 Features

- **Real-time Face Detection**: Automatically detects and tracks your eyes and mouth using AI-powered face landmarks
- **Interactive Masking**: Drag masks to reposition them, scroll to zoom in/out on each mask
- **Adjustable Feathering**: Control the blur/feather effect from 0-50 with a smooth slider
- **Dual Implementations**:
  - Desktop app with full performance and local processing
  - Web app for browser-based access (no installation needed)
- **Smooth Tracking**: Exponential moving average smoothing reduces jittery detection updates
- **Fallback Detection**: When AI models fail, cascades to Haar cascade fallback detection

## 🛠 Technical Stack

### Desktop Version (Python)
- **OpenCV**: Video capture and image processing
- **MediaPipe**: Face landmark detection (FaceLandmarker model)
- **tkinter**: GUI framework
- **Pillow (PIL)**: Image conversion and display

### Web Version (JavaScript)
- **MediaPipe FaceMesh**: Browser-based face landmark detection
- **Canvas API**: Real-time drawing and compositing
- **HTML5 Video**: Webcam access

## 📥 Installation

### Desktop Version

**Requirements**: Python 3.8+ and a webcam

1. Clone or download this repository
2. Create and activate a virtual environment:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

4. Run the application using one of these methods:
   ```powershell
   # PowerShell
   .\run.ps1
   
   # Or directly
   python app.py
   ```
   Or use `run.bat` from Command Prompt on Windows

### Web Version

No installation required! Simply:

1. Start a local web server:
   ```powershell
   python -m http.server 8000
   ```

2. Open your browser to:
   ```
   http://localhost:8000
   ```

3. Click "Start Webcam" and allow camera access

> **Important**: Use `http://localhost` (not `file://`) to ensure proper webcam access in the browser.

## 🚀 Usage

### Desktop Application

1. **Start**: Click "Start Webcam" to begin
2. **Interact with Masks**:
   - **Drag**: Click and drag any mask to reposition it
   - **Zoom**: Scroll wheel to scale masks larger/smaller
   - **Feather**: Adjust the "Mask Feather" slider (0-50) to control blur smoothness
3. **Stop**: Click "Stop Webcam" to end

### Web Application

Same controls as the desktop version:
- Click "Start Webcam" to begin
- Drag masks to adjust position
- Scroll to zoom masks in/out
- Use the feather slider to control blur intensity

## 🧠 How It Works

### Face Detection Pipeline

1. **Video Input**: Captures frames from your webcam
2. **Face Landmark Detection**: MediaPipe detects 468 facial landmarks
3. **Feature Box Calculation**: Eyes and mouth regions are computed from landmark positions:
   - **Left Eye**: 8 landmark points (indices 33, 160, 158, 133, 153, 144, 159, 145)
   - **Right Eye**: 8 landmark points (indices 362, 385, 387, 263, 373, 380, 386, 374)
   - **Mouth**: 22 landmark points (indices 61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308)

4. **Smoothing**: Exponential moving average (default α=0.65) smooths detection jitter
5. **Rendering**: Masks are drawn with:
   - Elliptical alpha masking for smooth edges
   - Gaussian blur for feathering effect
   - Blend compositing for natural blending

### Mask Rendering Algorithm

- Masks are extracted from the source frame and resized to the target region
- An elliptical alpha mask is created with Gaussian blur based on feather strength
- The mask is blended with the green background using pixel-wise alpha compositing
- This creates smooth, natural-looking blur effects

## ⚙️ Configuration

### Desktop App (app.py)

Adjustable parameters in the code:

```python
self.detection_interval = 1        # Run detection every N frames
self.landmark_smooth_alpha = 0.65  # Smoothing strength (0.0-1.0)
self.blur_strength = 15            # Default feather value (0-50)
self.detection_max_dim = 480       # Max dimension for detection (larger = slower)
```

### Model Download

The MediaPipe FaceLandmarker model (~60MB) is automatically downloaded on first run to `models/face_landmarker.task`.

## ⚡ Performance Optimization

### Desktop Version
- Detections run at configurable intervals (default: every frame)
- Frame downscaling before detection preserves accuracy while improving FPS
- Lazy landmark detection skipping reuses recent results

### Web Version
- MediaPipe runs efficiently in the browser
- Canvas rendering is optimized with minimal redraws
- Inference throttling prevents redundant predictions

## 💻 System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **CPU** | Dual-core 2GHz | Quad-core 2.5GHz+ |
| **RAM** | 2GB | 4GB+ |
| **Webcam** | 480p | 720p+ |
| **Python** | 3.8 | 3.10+ |
| **Browser** | Chrome/Edge | Latest Chrome/Edge |

## 🔧 Troubleshooting

### Desktop App Won't Start
- Ensure Python 3.8+ is installed: `python --version`
- Verify dependencies: `pip list`
- Reinstall dependencies: `pip install --upgrade -r requirements.txt`

### Camera Access Denied
- Check camera permissions in Windows/macOS/Linux settings
- Ensure no other application is using the webcam
- Restart the application

### Slow Performance
- Reduce `detection_max_dim` in the code (default: 480)
- Increase `detection_interval` to skip frames (default: 1)
- Close other applications to free resources

### Web App Camera Issues
- Ensure you're accessing via `http://localhost` (not `file://`)
- Check browser camera permissions
- Use Chrome/Edge (best compatibility)

### Poor Face Detection
- Ensure good lighting conditions
- Face should be clearly visible and ~200+ pixels wide
- Adjust distance from camera
- Try adjusting the lighting around your face

## 📁 File Structure

```
auge/
├── app.py                  # Desktop application (tkinter + OpenCV)
├── app.js                  # Web application (JavaScript/Canvas)
├── index.html              # Web application UI
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── SERVER_SETUP.md         # Server configuration guide
├── run.ps1                 # PowerShell launcher script
├── run.bat                 # Batch launcher script
└── models/
    └── face_landmarker.task  # MediaPipe model (auto-downloaded)
```

## 📜 License

This project uses:
- **MediaPipe** (Apache 2.0)
- **OpenCV** (Apache 2.0)
- **Pillow** (HPND)

## 🤝 Contributing

Improvements welcome! Potential enhancements:
- Additional mask shapes (rectangles, custom polygons)
- Real-time performance metrics overlay
- Snapshot/recording functionality
- Multiple face detection
- Custom mask colors and patterns

## 📝 Notes

- Webcam feed is processed locally; no data is transmitted
- Desktop app provides better performance for sustained usage
- Web app is ideal for quick access and testing
- Both versions process frames in real-time with minimal latency
- Press <kbd>Escape</kbd> to close the desktop app quickly (or use Stop Webcam button)

## 🆘 Support

For additional configuration details, refer to `SERVER_SETUP.md`.
