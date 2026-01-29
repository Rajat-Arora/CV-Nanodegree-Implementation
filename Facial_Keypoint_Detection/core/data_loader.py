"""
Data loading and transformation utilities for facial keypoint detection.

This module provides dataset classes and transformations for preparing
image and keypoint data for training and testing a CNN-based facial
keypoint detector.
"""

import os
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset


class FacialKeypointsDataset(Dataset):
    """Face Landmarks dataset for PyTorch."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Initialize the facial keypoints dataset.
        
        Args:
            csv_file (str): Path to the CSV file with annotations.
                           Expected format: image_name, x1, y1, x2, y2, ..., x68, y68
            root_dir (str): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                                          on a sample.
        """
        self.key_pts_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        """Return the total number of samples."""
        return len(self.key_pts_frame)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample to retrieve.
            
        Returns:
            dict: Dictionary containing 'image' and 'keypoints' keys.
        """
        image_name = os.path.join(self.root_dir,
                                  self.key_pts_frame.iloc[idx, 0])
        
        image = cv2.imread(image_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # If image has an alpha channel, remove it
        if len(image.shape) == 3 and image.shape[2] == 4:
            image = image[:, :, 0:3]
        
        # Extract keypoints from CSV
        key_pts = self.key_pts_frame.iloc[idx, 1:].values
        key_pts = key_pts.astype('float').reshape(-1, 2)
        
        sample = {'image': image, 'keypoints': key_pts}

        if self.transform:
            sample = self.transform(sample)

        return sample


# ======================== Transformation Classes ========================

class Normalize(object):
    """Convert image to grayscale and normalize color range.
    
    Transforms:
    - RGB image to grayscale
    - Color range from [0, 255] to [0, 1]
    - Keypoints from pixel coordinates to normalized range [-1, 1]
    
    The normalization assumes keypoints are centered around (100, 100)
    with standard deviation of 50 pixels.
    """

    def __call__(self, sample):
        """Apply normalization to a sample."""
        image, key_pts = sample['image'], sample['keypoints']
        
        image_copy = np.copy(image)
        key_pts_copy = np.copy(key_pts)

        # Convert image to grayscale
        image_copy = cv2.cvtColor(image_copy, cv2.COLOR_RGB2GRAY)
        
        # Normalize color range from [0, 255] to [0, 1]
        image_copy = image_copy / 255.0
        
        # Normalize keypoints to range [-1, 1]
        # Assumption: mean = 100, std = 50
        key_pts_copy = (key_pts_copy - 100) / 50.0

        return {'image': image_copy, 'keypoints': key_pts_copy}


class Rescale(object):
    """Rescale the image in a sample to a given size.
    
    Maintains aspect ratio and scales keypoints accordingly.

    Args:
        output_size (int or tuple): Desired output size.
            If int: smaller edge is matched to output_size.
            If tuple: (height, width)
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        """Apply rescaling to a sample."""
        image, key_pts = sample['image'], sample['keypoints']

        h, w = image.shape[:2]
        
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        # Resize image using interpolation
        img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Scale keypoints proportionally
        key_pts = key_pts * [new_w / w, new_h / h]

        return {'image': img, 'keypoints': key_pts}


class RandomCrop(object):
    """Randomly crop the image in a sample.

    Args:
        output_size (int or tuple): Desired output size.
            If int: square crop is made.
            If tuple: (height, width)
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        """Apply random cropping to a sample."""
        image, key_pts = sample['image'], sample['keypoints']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        # Crop image
        image = image[top: top + new_h, left: left + new_w]

        # Adjust keypoints relative to crop
        key_pts = key_pts - [left, top]

        return {'image': image, 'keypoints': key_pts}


class ToTensor(object):
    """Convert numpy arrays in sample to PyTorch tensors."""

    def __call__(self, sample):
        """Convert sample to tensors."""
        image, key_pts = sample['image'], sample['keypoints']
        
        # If image is 2D (grayscale), add channel dimension
        if len(image.shape) == 2:
            image = image.reshape(image.shape[0], image.shape[1], 1)
        
        # Convert from numpy format (H x W x C) to torch format (C x H x W)
        image = image.transpose((2, 0, 1))

        # Ensure float32 for model compatibility
        image = image.astype(np.float32)
        key_pts = key_pts.astype(np.float32)
        
        return {
            'image': torch.from_numpy(image),
            'keypoints': torch.from_numpy(key_pts)
        }
