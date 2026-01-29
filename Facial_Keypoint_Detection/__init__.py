"""
Facial Keypoint Detection - A CNN-based facial landmark detector.

This package provides tools for training and using a convolutional neural
network to detect 68 facial keypoints in images.

Modules:
    - models: CNN architecture definitions
    - data_loader: Dataset and transformation utilities
    - train: Training and evaluation pipelines
    - inference: Inference utilities and visualization
    - utils: Helper functions

Author: Rajat
Date: 2026
"""

from core.models import Net
from core.data_loader import (
    FacialKeypointsDataset,
    Normalize,
    Rescale,
    RandomCrop,
    ToTensor
)
from core.train import train_model, evaluate_model
from core.inference import (
    FaceDetector,
    KeypointDetector,
    detect_and_display_keypoints,
    visualize_keypoints,
    denormalize_keypoints
)

__version__ = '1.0.0'
__author__ = 'Rajat'

__all__ = [
    'Net',
    'FacialKeypointsDataset',
    'Normalize',
    'Rescale',
    'RandomCrop',
    'ToTensor',
    'train_model',
    'evaluate_model',
    'FaceDetector',
    'KeypointDetector',
    'detect_and_display_keypoints',
    'visualize_keypoints',
    'denormalize_keypoints'
]
