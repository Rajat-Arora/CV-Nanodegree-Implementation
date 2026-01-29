# Real-Time Facial Keypoint Detection App

Professional PyQt5-based GUI application for detecting facial keypoints in real-time using your webcam.

## Features

**Real-Time Detection**
- Live webcam feed with 10 FPS processing
- Automatic face detection using Haar Cascades
- Real time 68-keypoint facial landmark detection
- Mirror mode for natural viewing

**User-Friendly Interface**
- Clean, intuitive PyQt5 GUI
- Adjustable detection parameters
- Live statistics (FPS, faces detected, keypoints)
- GPU/CPU selection for optimization

**Customizable Settings**
- Camera selection (multiple cameras)
- Face detection scale factor adjustment
- Minimum neighbors parameter tuning
- Optional face bounding box display

**Performance Monitoring**
- Real-time FPS counter
- Face detection count
- Processing statistics
- Error reporting

---

## Installation

### Step 1: Complete the installation step from the project root directory

### Step 2: Download Pre-trained Model (Optional)

Train the model, using 03_train_model.py script which saved the trained model as:
```
saved_models/keypoints_model_best.pt
```

The app will work in **demo mode** without a trained model (showing random keypoints).

---

## Using the Application

### 1. **Start the App**
   - Launch `python app/gui.py`
   - Wait for the GUI to load

### 2. **Select Camera**
   - Use the "Camera:" dropdown to select your webcam
   - Usually "Camera 0" is the default camera

### 3. **Click "Start Detection"**
   - Green button in the left panel
   - Webcam feed will appear
   - Faces will be outlined, keypoints shown as yellow dots

### 4. **Adjust Parameters** (Optional)
   - **Face Detection Scale**: Adjust sensitivity (1.00-1.50)
   - **Min Neighbors**: Reduce for more detections, increase for fewer false positives
   - **Show Face Bounding Box**: Toggle face outlines

### 5. **Monitor Statistics**
   - FPS: Frames per second (target: 25-30)
   - Faces Detected: Number of faces in current frame
   - Keypoints: Total keypoints detected

### 6. **Use GPU** (Optional)
   - If you have NVIDIA GPU, check "Use GPU (CUDA)"
   - Significantly speeds up processing

### 7. **Stop Detection**
   - Click red "Stop Detection" button

---

## System Requirements

### Minimum
- Python 3.8+
- 4GB RAM
- Webcam (USB or built-in)
- CPU: Intel i5 or equivalent

### Recommended
- Python 3.8+
- 8GB+ RAM
- NVIDIA GPU (CUDA capable)
- High-quality webcam (1080p+)

---

## Troubleshooting

### Issue: Camera not detected
**Solution**: 
```bash
# List available cameras
python << 'EOF'
import cv2
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera {i}: Available")
        cap.release()
    else:
        print(f"Camera {i}: Not available")
EOF
```

### Issue: Low FPS / Slow processing
**Solutions**:
1. Use GPU if available (check "Use GPU" checkbox)
2. Reduce webcam resolution in camera settings
3. Lower face detection scale factor
4. Close other applications

### Issue: "Model not found"
**Solution**:
- Train a model first: `python example_usage.py`
- Or download pre-trained model to `saved_models/keypoints_model_best.pt`
- App works in demo mode without a trained model

### Issue: PyQt5 errors
**Solution**:
```bash
pip install --upgrade PyQt5
```

### Issue: "Failed to open camera"
**Solution**:
```bash
# Check permissions (Linux)
ls -l /dev/video*
# May need to add user to video group
sudo usermod -a -G video $USER
```

---

## Advanced Usage

### Custom Model Path

Edit `app/gui.py` and change:
```python
model_path = Path(__file__).parent.parent / 'saved_models' / 'keypoints_model_best.pt'
```

to:
```python
model_path = Path('/your/custom/path/to/model.pt')
```

### Batch Processing Videos

Use `webcam_detector.py` directly:
```python
import cv2
from models import Net
from app.webcam_detector import WebcamKeypointDetector

model = Net()
detector = WebcamKeypointDetector(model, 'path/to/model.pt')

cap = cv2.VideoCapture('video.mp4')
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    annotated, faces, keypoints = detector.process_frame(frame)
    cv2.imshow('Keypoints', annotated)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## Performance Tips

### For Better FPS
1. **Use GPU**: Check the CUDA checkbox if available
2. **Lower Resolution**: Reduce camera resolution
3. **Adjust Detection**: Increase scale factor (less sensitive)
4. **Close Other Apps**: Free up system resources

### For Better Accuracy
1. **Good Lighting**: Ensure faces are well-lit
2. **Camera Distance**: Keep faces 1-2 meters away
3. **Frontal Pose**: Ensure faces are mostly frontal
4. **Quality Camera**: Higher resolution helps

---

## Keypoint Visualization

The 68 facial keypoints represent:

```
Jaw (0-16): Chin outline and jawline
Eyebrows (17-26): Left and right eyebrows
Nose (27-35): Nose outline
Eyes (36-47): Left and right eye outlines
Mouth (48-67): Mouth outline and interior
```

Yellow dots: Detected keypoints
Green box: Face bounding box
FPS counter: Real-time performance

---

## Architecture

### Components

```
GUI (gui.py)
├── Video Display
├── Controls
│   ├── Camera Selection
│   ├── Start/Stop Buttons
│   └── Parameter Adjustments
└── Statistics Display

Detection Engine (webcam_detector.py)
├── Face Detection (Haar Cascade)
├── Preprocessing
├── Model Inference
└── Visualization

Model (models.py)
└── 4-Layer CNN (pre-trained)
```

### Threading Model

- Main Thread: PyQt5 GUI
- Worker Thread: Webcam capture and processing
- Prevents GUI freezing during intensive computation

---

## Keyboard Shortcuts

Inside the app window (when focusing on video):
- **Space**: Pause/Resume
- **Q or Escape**: Quit application
- **S**: Save current frame
- **T**: Toggle face bounding box

(Note: These can be added to the GUI in future versions)

---

## Customization Examples

### Change Color Scheme

Edit `gui.py` and modify button colors:
```python
self.start_btn.setStyleSheet("background-color: #YOUR_COLOR; ...")
```

### Add New Detection Parameters

Add to `__init_ui__()`:
```python
self.threshold_slider = QSlider(Qt.Horizontal)
control_layout.addWidget(self.threshold_slider)
```

### Save Detections

Modify `update_frame()` to save frames:
```python
cv2.imwrite(f'output/frame_{frame_count}.jpg', frame)
```

---

