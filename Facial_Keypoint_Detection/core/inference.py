"""
Inference utilities for facial keypoint detection.

This module provides functions for detecting facial keypoints in images
using trained models and Haar Cascade classifiers for face detection.
"""

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path


class FaceDetector:
    """Face detection using Haar Cascade classifiers."""
    
    def __init__(self, cascade_path='detector_architectures/haarcascade_frontalface_default.xml'):
        """
        Initialize face detector.
        
        Args:
            cascade_path (str): Path to Haar Cascade XML file.
        """
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            raise FileNotFoundError(f"Cascade classifier not found at {cascade_path}")
    
    def detect(self, image, scale_factor=1.2, min_neighbors=2):
        """
        Detect faces in an image.
        
        Args:
            image (np.ndarray): Input image (BGR format).
            scale_factor (float): Scale factor for cascade classifier.
            min_neighbors (int): Minimum neighbors for cascade classifier.
            
        Returns:
            np.ndarray: Array of detected faces (x, y, w, h).
        """
        faces = self.face_cascade.detectMultiScale(
            image, scaleFactor=scale_factor, minNeighbors=min_neighbors
        )
        return faces


class KeypointDetector:
    """Facial keypoint detector using pre-trained CNN."""
    
    def __init__(self, model, model_path, device='cpu'):
        """
        Initialize keypoint detector.
        
        Args:
            model (nn.Module): The CNN model architecture.
            model_path (str): Path to saved model weights.
            device (str): Device to use ('cpu' or 'cuda').
        """
        self.model = model.to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        self.device = device
    
    def preprocess_face(self, face_roi, size=224, padding=0):
        """
        Preprocess a face region for keypoint detection.
        
        Args:
            face_roi (np.ndarray): Face region of interest (BGR).
            size (int): Target image size (224x224 recommended).
            padding (int): Padding to add around face.
            
        Returns:
            torch.Tensor: Preprocessed tensor (1, 1, 224, 224).
        """
        # Add padding if specified
        if padding > 0:
            h, w = face_roi.shape[:2]
            face_roi = cv2.copyMakeBorder(
                face_roi, padding, padding, padding, padding,
                cv2.BORDER_CONSTANT, value=[0, 0, 0]
            )
        
        # Convert to grayscale
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        # Normalize to [0, 1]
        gray = gray / 255.0
        
        # Resize to target size
        gray = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
        
        # Convert to tensor with shape (1, 1, H, W)
        tensor = torch.from_numpy(gray).float()
        tensor = tensor.unsqueeze(0).unsqueeze(0)
        
        return tensor
    
    def predict(self, face_tensor):
        """
        Predict keypoints for a face tensor.
        
        Args:
            face_tensor (torch.Tensor): Preprocessed face tensor.
            
        Returns:
            np.ndarray: Predicted keypoints (68, 2).
        """
        with torch.no_grad():
            face_tensor = face_tensor.to(self.device)
            output = self.model(face_tensor)
            output = output.view(output.size(0), 68, -1)
            keypoints = output[0].cpu().numpy()
        
        return keypoints


def denormalize_keypoints(keypoints, scale=1.0):
    """
    Denormalize keypoints from [-1, 1] range to pixel coordinates.
    
    Args:
        keypoints (np.ndarray): Normalized keypoints (68, 2).
        scale (float): Scale factor.
        
    Returns:
        np.ndarray: Denormalized keypoints.
    """
    # Original normalization: (kpts - 100) / 50
    # Reverse: kpts * 50 + 100
    return keypoints * 50.0 * scale + 100.0


def visualize_keypoints(image, keypoints, keypoint_format='denormalized',
                       marker_size=50, marker_color='magenta', show=True):
    """
    Visualize keypoints on an image.
    
    Args:
        image (np.ndarray): Image to visualize on.
        keypoints (np.ndarray): Keypoints (68, 2).
        keypoint_format (str): 'normalized' or 'denormalized'.
        marker_size (int): Size of markers.
        marker_color (str): Color of markers.
        show (bool): Whether to display the image.
        
    Returns:
        matplotlib.figure.Figure: Figure object.
    """
    # Convert to grayscale if needed for visualization
    if len(image.shape) == 3:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(image, cmap='gray')
    
    if keypoint_format == 'normalized':
        keypoints = denormalize_keypoints(keypoints)
    
    # Clip keypoints to image bounds
    h, w = image.shape[:2]
    keypoints[:, 0] = np.clip(keypoints[:, 0], 0, w)
    keypoints[:, 1] = np.clip(keypoints[:, 1], 0, h)
    
    plt.scatter(keypoints[:, 0], keypoints[:, 1], 
               s=marker_size, marker='.', c=marker_color)
    plt.axis('off')
    
    if show:
        plt.show()
    
    return fig


def detect_and_display_keypoints(image_path, model, face_detector, 
                                 keypoint_detector, model_output_format='normalized'):
    """
    Detect faces and keypoints in an image and display results.
    
    Args:
        image_path (str): Path to image file.
        model (nn.Module): CNN model (for reference).
        face_detector (FaceDetector): Face detector object.
        keypoint_detector (KeypointDetector): Keypoint detector object.
        model_output_format (str): 'normalized' or 'denormalized'.
        
    Returns:
        tuple: (image, faces, all_keypoints)
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Detect faces
    faces = face_detector.detect(image)
    
    if len(faces) == 0:
        print("No faces detected in the image")
        return image_rgb, faces, []
    
    print(f"Detected {len(faces)} face(s)")
    
    # Detect keypoints for each face
    all_keypoints = []
    for i, (x, y, w, h) in enumerate(faces):
        face_roi = image[y:y+h, x:x+w]
        
        # Preprocess and predict
        face_tensor = keypoint_detector.preprocess_face(face_roi, padding=40)
        keypoints = keypoint_detector.predict(face_tensor)
        all_keypoints.append(keypoints)
    
    # Visualize
    fig, axes = plt.subplots(1, len(faces), figsize=(6*len(faces), 6))
    if len(faces) == 1:
        axes = [axes]
    
    for i, (keypoints, (x, y, w, h)) in enumerate(zip(all_keypoints, faces)):
        face_roi = image[y:y+h, x:x+w]
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        ax = axes[i]
        ax.imshow(gray, cmap='gray')
        
        if model_output_format == 'normalized':
            keypoints = denormalize_keypoints(keypoints)
        
        ax.scatter(keypoints[:, 0], keypoints[:, 1],
                  s=50, marker='.', c='magenta')
        ax.set_title(f'Face {i+1}')
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return image_rgb, faces, all_keypoints
