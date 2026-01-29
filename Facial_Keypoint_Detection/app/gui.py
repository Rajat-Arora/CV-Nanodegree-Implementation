"""
PyQt5-based GUI for real-time facial keypoint detection.

This application provides a user-friendly interface for detecting facial
landmarks in real-time using a webcam.
"""

import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QCheckBox, QSpinBox, QSlider,
    QStatusBar, QFrame
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QImage, QPixmap, QFont
from pathlib import Path
import torch
import os
import sys

# Add parent directory to path to import project modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.models import Net
from app.webcam_detector import WebcamKeypointDetector


class WebcamThread(QThread):
    """Separate thread for webcam processing to prevent UI freezing."""
    
    frame_signal = pyqtSignal(np.ndarray, int, int, list)  # frame, num_faces, fps, keypoints
    error_signal = pyqtSignal(str)
    
    def __init__(self, detector, camera_id=0, use_cuda=False):
        super().__init__()
        self.detector = detector
        self.camera_id = camera_id
        self.running = False
        self.use_cuda = use_cuda
        
    def run(self):
        """Run the webcam capture loop."""
        cap = cv2.VideoCapture(self.camera_id)
        
        if not cap.isOpened():
            self.error_signal.emit("Failed to open camera")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.running = True
        
        try:
            while self.running:
                ret, frame = cap.read()
                
                if not ret:
                    self.error_signal.emit("Failed to read frame from camera")
                    break
                
                # Flip for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Process frame
                try:
                    annotated, faces, keypoints = self.detector.process_frame(frame)
                    num_faces = len(faces)
                    fps = int(self.detector.fps)
                    self.frame_signal.emit(annotated, num_faces, fps, keypoints)
                except Exception as e:
                    self.error_signal.emit(f"Processing error: {str(e)}")
        
        finally:
            cap.release()
    
    def stop(self):
        """Stop the webcam thread."""
        self.running = False
        self.wait()


class FacialKeypointApp(QMainWindow):
    """Main application window for facial keypoint detection."""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.setup_detector()
        self.webcam_thread = None
        
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("Facial Keypoint Detection - Real-time Detector")
        self.setGeometry(100, 100, 1400, 900)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout()
        
        # Video display area (left side)
        video_layout = QVBoxLayout()
        
        # Video frame
        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("border: 2px solid black; background-color: black;")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setScaledContents(False)
        video_layout.addWidget(self.video_label)
        
        # Controls (right side)
        control_layout = QVBoxLayout()
        
        # Title
        title_label = QLabel("Facial Keypoint Detector")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title_label.setFont(title_font)
        control_layout.addWidget(title_label)
        
        # Status
        self.status_label = QLabel("Status: Initializing...")
        control_layout.addWidget(self.status_label)
        
        # Separator
        sep1 = QFrame()
        sep1.setFrameShape(QFrame.HLine)
        control_layout.addWidget(sep1)
        
        # Camera selection
        camera_label = QLabel("Camera:")
        control_layout.addWidget(camera_label)
        self.camera_combo = QComboBox()
        self.camera_combo.addItems(["Camera 0", "Camera 1", "Camera 2"])
        control_layout.addWidget(self.camera_combo)
        
        # Start/Stop buttons
        button_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("Start Detection")
        self.start_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 8px;")
        self.start_btn.clicked.connect(self.start_detection)
        button_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("Stop Detection")
        self.stop_btn.setStyleSheet("background-color: #f44336; color: white; font-weight: bold; padding: 8px;")
        self.stop_btn.clicked.connect(self.stop_detection)
        self.stop_btn.setEnabled(False)
        button_layout.addWidget(self.stop_btn)
        
        control_layout.addLayout(button_layout)
        
        # Separator
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.HLine)
        control_layout.addWidget(sep2)
        
        # Detection parameters
        params_label = QLabel("Detection Parameters:")
        params_font = QFont()
        params_font.setBold(True)
        params_label.setFont(params_font)
        control_layout.addWidget(params_label)
        
        # Scale factor
        scale_label = QLabel("Face Detection Scale:")
        control_layout.addWidget(scale_label)
        self.scale_slider = QSlider(Qt.Horizontal)
        self.scale_slider.setMinimum(100)
        self.scale_slider.setMaximum(150)
        self.scale_slider.setValue(120)
        self.scale_slider.setTickPosition(QSlider.TicksBelow)
        self.scale_slider.setTickInterval(5)
        self.scale_value_label = QLabel("1.20")
        scale_layout = QHBoxLayout()
        scale_layout.addWidget(self.scale_slider)
        scale_layout.addWidget(self.scale_value_label)
        control_layout.addLayout(scale_layout)
        self.scale_slider.valueChanged.connect(self.update_scale_label)
        
        # Min neighbors
        neighbors_label = QLabel("Min Neighbors:")
        control_layout.addWidget(neighbors_label)
        self.neighbors_spinbox = QSpinBox()
        self.neighbors_spinbox.setMinimum(1)
        self.neighbors_spinbox.setMaximum(10)
        self.neighbors_spinbox.setValue(2)
        control_layout.addWidget(self.neighbors_spinbox)
        
        # Options
        options_label = QLabel("Options:")
        options_font = QFont()
        options_font.setBold(True)
        options_label.setFont(options_font)
        control_layout.addWidget(options_label)
        
        self.face_box_check = QCheckBox("Show Face Bounding Box")
        self.face_box_check.setChecked(True)
        control_layout.addWidget(self.face_box_check)
        
        self.cuda_check = QCheckBox("Use GPU (CUDA)")
        self.cuda_check.setEnabled(torch.cuda.is_available())
        control_layout.addWidget(self.cuda_check)
        
        # Info section
        sep3 = QFrame()
        sep3.setFrameShape(QFrame.HLine)
        control_layout.addWidget(sep3)
        
        info_label = QLabel("Information:")
        info_font = QFont()
        info_font.setBold(True)
        info_label.setFont(info_font)
        control_layout.addWidget(info_label)
        
        self.info_label = QLabel(
            f"Model: Facial Keypoint CNN\n"
            f"Input: Grayscale 224×224\n"
            f"Output: 68 keypoints\n"
            f"CUDA Available: {torch.cuda.is_available()}"
        )
        self.info_label.setStyleSheet("background-color: #f0f0f0; padding: 10px; border-radius: 5px;")
        control_layout.addWidget(self.info_label)
        
        # Statistics
        sep4 = QFrame()
        sep4.setFrameShape(QFrame.HLine)
        control_layout.addWidget(sep4)
        
        stats_label = QLabel("Statistics:")
        stats_font = QFont()
        stats_font.setBold(True)
        stats_label.setFont(stats_font)
        control_layout.addWidget(stats_label)
        
        self.stats_label = QLabel(
            "FPS: 0\n"
            "Faces Detected: 0\n"
            "Keypoints: 0"
        )
        self.stats_label.setStyleSheet("background-color: #e8f5e9; padding: 10px; border-radius: 5px;")
        control_layout.addWidget(self.stats_label)
        
        # Add stretch to push everything to top
        control_layout.addStretch()
        
        # Add layouts to main
        left_frame = QWidget()
        left_frame.setLayout(video_layout)
        
        right_frame = QWidget()
        right_frame.setLayout(control_layout)
        right_frame.setMaximumWidth(350)
        
        main_layout.addWidget(left_frame, 2)
        main_layout.addWidget(right_frame, 1)
        
        central_widget.setLayout(main_layout)
        
        # Status bar
        self.statusBar().showMessage("Ready")
        
    def setup_detector(self):
        """Initialize the detector."""
        try:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model = Net()
            
            model_path = Path(__file__).parent.parent / 'saved_models' / 'keypoints_model_best.pt'
            cascade_path = Path(__file__).parent.parent / 'detector_architectures' / 'haarcascade_frontalface_default.xml'
            
            self.detector = WebcamKeypointDetector(
                model=model,
                model_path=str(model_path),
                cascade_path=str(cascade_path),
                device=device
            )
            self.status_label.setText("Status: Ready to detect")
        except Exception as e:
            self.status_label.setText(f"Status: Error - {str(e)}")
            self.detector = None
    
    def start_detection(self):
        """Start real-time detection."""
        if self.detector is None:
            self.status_label.setText("Status: Detector not initialized")
            return
        
        camera_id = int(self.camera_combo.currentText().split()[-1])
        device = 'cuda' if (self.cuda_check.isChecked() and torch.cuda.is_available()) else 'cpu'
        self.detector.device = device
        self.detector.model = self.detector.model.to(device)
        
        self.webcam_thread = WebcamThread(self.detector, camera_id, device == 'cuda')
        self.webcam_thread.frame_signal.connect(self.update_frame)
        self.webcam_thread.error_signal.connect(self.handle_error)
        self.webcam_thread.start()
        
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.camera_combo.setEnabled(False)
        self.cuda_check.setEnabled(False)
        self.status_label.setText("Status: Detecting...")
        self.statusBar().showMessage("Detection running...")
    
    def stop_detection(self):
        """Stop real-time detection."""
        if self.webcam_thread:
            self.webcam_thread.stop()
        
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.camera_combo.setEnabled(True)
        self.cuda_check.setEnabled(True)
        self.status_label.setText("Status: Stopped")
        self.statusBar().showMessage("Detection stopped")
    
    def update_frame(self, frame, num_faces, fps, keypoints):
        """Update the video display."""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        
        # Convert to QImage
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Scale to fit label
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaledToWidth(640, Qt.SmoothTransformation)
        self.video_label.setPixmap(scaled_pixmap)
        
        # Update statistics
        total_keypoints = sum(len(kpts) for kpts in keypoints) if keypoints else 0
        self.stats_label.setText(
            f"FPS: {fps}\n"
            f"Faces Detected: {num_faces}\n"
            f"Keypoints: {total_keypoints * 2}"
        )
    
    def handle_error(self, error_msg):
        """Handle errors from webcam thread."""
        self.status_label.setText(f"Status: Error - {error_msg}")
        self.stop_detection()
    
    def update_scale_label(self, value):
        """Update scale factor label."""
        scale = value / 100.0
        self.scale_value_label.setText(f"{scale:.2f}")
    
    def closeEvent(self, event):
        """Clean up on window close."""
        if self.webcam_thread and self.webcam_thread.running:
            self.webcam_thread.stop()
        event.accept()


def main():
    """Main entry point."""
    app = QApplication(sys.argv)
    window = FacialKeypointApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
