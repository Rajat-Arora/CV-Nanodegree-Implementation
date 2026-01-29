"""
Convolutional Neural Network architecture for facial keypoint detection.

This module defines a multi-layer CNN designed to predict 68 facial keypoints
from grayscale face images. The architecture uses convolutional layers,
max pooling, and fully connected layers with dropout regularization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    """CNN for facial keypoint detection.
    
    Architecture:
    - 4 convolutional blocks (conv + ReLU + max pooling)
    - 2 fully connected layers with dropout
    - Output: 136 values (68 keypoints × 2 coordinates)
    
    Input shape: (batch_size, 1, 224, 224)  # Grayscale 224×224 image
    Output shape: (batch_size, 136)         # 68 keypoints (x, y)
    """

    def __init__(self):
        """Initialize the CNN architecture."""
        super(Net, self).__init__()
        
        # Convolutional block 1
        # Input: (1, 224, 224)
        # Output after conv: (32, 220, 220)
        # Output after pool: (32, 110, 110)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=0)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Convolutional block 2
        # Input: (32, 110, 110)
        # Output after conv: (64, 106, 106)
        # Output after pool: (64, 53, 53)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=0)
        
        # Convolutional block 3
        # Input: (64, 53, 53)
        # Output after conv: (128, 49, 49)
        # Output after pool: (128, 24, 24)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, padding=0)
        
        # Convolutional block 4
        # Input: (128, 24, 24)
        # Output after conv: (256, 20, 20)
        # Output after pool: (256, 10, 10)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=5, padding=0)
        
        # Fully connected layers
        # Flattened size: 256 × 10 × 10 = 25600
        self.fc1 = nn.Linear(256 * 10 * 10, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc_out = nn.Linear(512, 136)  # 68 keypoints × 2 coordinates
        
        # Regularization
        self.dropout = nn.Dropout(p=0.3)
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 224, 224)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 136)
        """
        # Convolutional blocks with ReLU activation and max pooling
        x = self.pool(F.relu(self.conv1(x)))  # -> (batch, 32, 110, 110)
        x = self.pool(F.relu(self.conv2(x)))  # -> (batch, 64, 53, 53)
        x = self.pool(F.relu(self.conv3(x)))  # -> (batch, 128, 24, 24)
        x = self.pool(F.relu(self.conv4(x)))  # -> (batch, 256, 10, 10)
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)  # -> (batch, 25600)
        
        # Fully connected layers with dropout
        x = self.dropout(x)
        x = self.dropout(F.relu(self.fc1(x)))  # -> (batch, 1024)
        x = self.dropout(F.relu(self.fc2(x)))  # -> (batch, 512)
        x = self.fc_out(x)  # -> (batch, 136)
        
        return x
