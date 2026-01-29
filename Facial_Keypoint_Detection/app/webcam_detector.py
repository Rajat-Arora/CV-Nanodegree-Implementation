"""
Webcam-based facial keypoint detection engine.

This module handles real-time facial keypoint detection using a pre-trained
CNN model and webcam input.
"""

import cv2
import numpy as np
import torch
from pathlib import Path
import time


class WebcamKeypointDetector:
    """Real-time facial keypoint detector using webcam."""
    
    def __init__(self, model, model_path='saved_models/keypoints_model_best.pt', 
                 cascade_path='detector_architectures/haarcascade_frontalface_default.xml',
                 device='cpu'):
        """
        Initialize the detector.
        
        Args:
            model: PyTorch model class (not initialized)
            model_path: Path to saved model weights
            cascade_path: Path to Haar Cascade XML file
            device: Device to use ('cpu' or 'cuda')
        """
        self.device = device
        self.model = model.to(device)
        
        # Try to load model weights
        try:
            checkpoint = torch.load(model_path, map_location=device)
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            self.model.eval()
            self.model_loaded = True
            print(f"Model loaded from {model_path}")
        except FileNotFoundError:
            self.model_loaded = False
            print(f"Model not found at {model_path}")
            print("Using untrained model for demonstration")
        except Exception as e:
            self.model_loaded = False
            print(f"Error loading model: {str(e)}")
            print("Using untrained model for demonstration")
        
        # Load face cascade
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            raise FileNotFoundError(f"Cascade classifier not found at {cascade_path}")
        
        print(f"Face detector loaded from {cascade_path}")
        
        # Statistics
        self.fps = 0
        self.frame_count = 0
        self.last_time = time.time()
        
    def detect_faces(self, image, scale_factor=1.2, min_neighbors=2):
        """Detect faces in image using Haar Cascade."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=scale_factor, minNeighbors=min_neighbors,
            minSize=(30, 30)
        )
        return faces
    
    def preprocess_face(self, face_roi, size=224, padding=40):
        """Preprocess face region for model input."""
        # Add padding
        h, w = face_roi.shape[:2]
        padded = cv2.copyMakeBorder(
            face_roi, padding, padding, padding, padding,
            cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )
        
        # Convert to grayscale
        gray = cv2.cvtColor(padded, cv2.COLOR_BGR2GRAY)
        
        # Normalize
        gray = gray / 255.0
        
        # Resize
        gray = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
        
        # Convert to tensor (1, 1, H, W)
        tensor = torch.from_numpy(gray).float()
        tensor = tensor.unsqueeze(0).unsqueeze(0)
        
        return tensor
    
    def predict_keypoints(self, face_tensor):
        """Predict keypoints for a face tensor."""
        if not self.model_loaded:
            # Return random keypoints for demo
            return np.random.randn(68, 2) * 30 + 100
        
        with torch.no_grad():
            face_tensor = face_tensor.to(self.device)
            output = self.model(face_tensor)
            output = output.view(output.size(0), 68, -1)
            keypoints = output[0].cpu().numpy()
        
        return keypoints
    
    def denormalize_keypoints(self, keypoints, face_shape, padding=40):
        """Convert normalized keypoints to pixel coordinates."""
        h, w = face_shape[:2]
        
        # Denormalize from [-1, 1] to pixel coordinates
        # Model was trained with: (keypoints - 100) / 50.0
        # So we reverse: keypoints * 50.0 + 100.0
        keypoints_denorm = keypoints * 50.0 + 100.0
        
        # Adjust for padding that was added during preprocessing
        # The padding shifts the coordinates by 40 pixels
        keypoints_denorm[:, 0] -= padding
        keypoints_denorm[:, 1] -= padding
        
        return keypoints_denorm
    
    def draw_keypoints(self, image, faces, all_keypoints, draw_face_box=True):
        """Draw keypoints and face boxes on image."""
        annotated = image.copy()
        
        for idx, (x, y, w, h) in enumerate(faces):
            # Draw face bounding box
            if draw_face_box:
                cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            if idx < len(all_keypoints):
                keypoints = all_keypoints[idx]
                
                # Denormalize keypoints from normalized space
                keypoints_pixel = self.denormalize_keypoints(
                    keypoints.copy(), (h, w), padding=40
                )
                
                # The keypoints are now in a coordinate system relative to the padded face
                # which was resized to 224x224. We need to scale them back to the original face size.
                # The padded face was (w+80, h+80) before resize, so:
                padded_w = w + 80
                padded_h = h + 80
                scale_x = padded_w / 224.0
                scale_y = padded_h / 224.0
                
                kpts_scaled = keypoints_pixel * [scale_x, scale_y]
                kpts_scaled[:, 0] += x
                kpts_scaled[:, 1] += y
                
                # Draw keypoints
                for kpt in kpts_scaled:
                    pt = tuple(map(int, kpt))
                    if 0 <= pt[0] < annotated.shape[1] and 0 <= pt[1] < annotated.shape[0]:
                        cv2.circle(annotated, pt, 3, (0, 255, 255), -1)
        
        return annotated
    
    def process_frame(self, frame):
        """Process a single frame: detect faces and predict keypoints."""
        faces = self.detect_faces(frame)
        all_keypoints = []
        
        for x, y, w, h in faces:
            face_roi = frame[y:y+h, x:x+w]
            
            # Preprocess
            face_tensor = self.preprocess_face(face_roi, padding=40)
            
            # Predict
            keypoints = self.predict_keypoints(face_tensor)
            all_keypoints.append(keypoints)
        
        # Draw
        annotated = self.draw_keypoints(frame, faces, all_keypoints)
        
        # Update FPS
        self.frame_count += 1
        current_time = time.time()
        if current_time - self.last_time >= 1.0:
            self.fps = self.frame_count / (current_time - self.last_time)
            self.frame_count = 0
            self.last_time = current_time
        
        # Add FPS text
        cv2.putText(annotated, f'FPS: {self.fps:.1f}', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated, f'Faces: {len(faces)}', (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        if not self.model_loaded:
            cv2.putText(annotated, 'Demo Mode (No Model)', (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return annotated, faces, all_keypoints
