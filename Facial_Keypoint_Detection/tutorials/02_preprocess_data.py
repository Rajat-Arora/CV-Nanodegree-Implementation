"""
Script 2: Preprocess Facial Keypoint Data

This script preprocesses the facial keypoint dataset by:
- Loading raw data
- Normalizing images and keypoints
- Applying transformations
- Creating PyTorch data loaders
- Saving preprocessed samples for inspection

Usage:
    python 02_preprocess_data.py
"""

import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import cv2

# Add parent directory to path to import core modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import our custom data loader
from core.data_loader import FacialKeypointsDataset, Rescale, RandomCrop, Normalize, ToTensor

# Configuration
DATA_DIR = 'data'
TRAINING_CSV = os.path.join(DATA_DIR, 'training_frames_keypoints.csv')
TEST_CSV = os.path.join(DATA_DIR, 'test_frames_keypoints.csv')
BATCH_SIZE = 16
IMAGE_SIZE = 224


def create_transforms(training=True):
    """Create data transformation pipeline."""
    if training:
        # Training transforms with augmentation
        return transforms.Compose([
            # Force a consistent square size before random crop
            Rescale((250, 250)),
            RandomCrop(IMAGE_SIZE),
            Normalize(),
            ToTensor()
        ])
    else:
        # Test transforms without augmentation (fixed square to avoid batch size mismatch)
        return transforms.Compose([
            Rescale((IMAGE_SIZE, IMAGE_SIZE)),
            Normalize(),
            ToTensor()
        ])


def create_data_loaders(batch_size=16):
    """Create PyTorch data loaders for training and validation."""
    print(f"\n{'='*60}")
    print(f"CREATING DATA LOADERS")
    print(f"{'='*60}")
    
    # Check if data exists
    if not os.path.exists(TRAINING_CSV):
        print(f"Error: Training CSV not found at {TRAINING_CSV}")
        print(f"\nPlease run 01_load_and_visualize.py first to set up data")
        return None, None
    
    # Create datasets
    print(f"\nLoading training dataset...")
    train_transform = create_transforms(training=True)
    train_dataset = FacialKeypointsDataset(
        csv_file=TRAINING_CSV,
        root_dir=os.path.join(DATA_DIR, 'training'),
        transform=train_transform
    )
    print(f"Training samples: {len(train_dataset)}")
    
    print(f"\nLoading test dataset...")
    test_transform = create_transforms(training=False)
    test_dataset = FacialKeypointsDataset(
        csv_file=TEST_CSV,
        root_dir=os.path.join(DATA_DIR, 'test'),
        transform=test_transform
    )
    print(f"Test samples: {len(test_dataset)}")
    
    # Create data loaders
    print(f"\nCreating data loaders (batch_size={batch_size})...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  # Set to 0 to avoid multiprocessing issues
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    print(f"Training batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    return train_loader, test_loader


def visualize_batch(data_loader, num_samples=4, save_path='preprocessed_samples.png'):
    """Visualize a batch of preprocessed samples."""
    print(f"\nVisualizing preprocessed samples...")
    
    # Get one batch
    batch = next(iter(data_loader))
    images = batch['image']
    keypoints = batch['keypoints']
    
    # Select samples to display
    num_samples = min(num_samples, len(images))
    
    # Create subplot grid
    rows = int(np.ceil(num_samples / 2))
    cols = 2
    fig, axes = plt.subplots(rows, cols, figsize=(12, 6 * rows))
    axes = axes.flatten() if num_samples > 1 else [axes]
    
    for idx in range(num_samples):
        # Get image and keypoints
        img = images[idx].squeeze().numpy()  # Remove channel dim, convert to numpy
        kpts = keypoints[idx].numpy()
        
        # Denormalize image for visualization
        img = img * 255.0
        img = img.astype(np.uint8)
        
        # Denormalize keypoints (reverse the normalization: (y - 100) / 50)
        kpts = kpts * 50.0 + 100.0
        kpts = kpts.reshape(-1, 2)
        
        # Display
        plt.sca(axes[idx])
        plt.imshow(img, cmap='gray')
        plt.scatter(kpts[:, 0], kpts[:, 1], 
                   s=20, c='lime', marker='o', edgecolors='yellow', linewidth=1)
        plt.title(f'Preprocessed Sample {idx + 1}', fontsize=12, fontweight='bold')
        plt.axis('off')
    
    # Hide unused subplots
    for idx in range(num_samples, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to '{save_path}'")
    plt.show()


def inspect_batch_shapes(train_loader, test_loader):
    """Inspect and print batch shapes."""
    print(f"\n{'='*60}")
    print(f"BATCH SHAPE INSPECTION")
    print(f"{'='*60}")
    
    # Training batch
    train_batch = next(iter(train_loader))
    print(f"\nTraining Batch:")
    print(f"   Image shape: {train_batch['image'].shape}")
    print(f"   Keypoints shape: {train_batch['keypoints'].shape}")
    print(f"   Image dtype: {train_batch['image'].dtype}")
    print(f"   Keypoints dtype: {train_batch['keypoints'].dtype}")
    print(f"   Image range: [{train_batch['image'].min():.3f}, {train_batch['image'].max():.3f}]")
    print(f"   Keypoints range: [{train_batch['keypoints'].min():.3f}, {train_batch['keypoints'].max():.3f}]")
    
    # Test batch
    test_batch = next(iter(test_loader))
    print(f"\nTest Batch:")
    print(f"   Image shape: {test_batch['image'].shape}")
    print(f"   Keypoints shape: {test_batch['keypoints'].shape}")
    print(f"   Image dtype: {test_batch['image'].dtype}")
    print(f"   Keypoints dtype: {test_batch['keypoints'].dtype}")
    print(f"   Image range: [{test_batch['image'].min():.3f}, {test_batch['image'].max():.3f}]")
    print(f"   Keypoints range: [{test_batch['keypoints'].min():.3f}, {test_batch['keypoints'].max():.3f}]")
    
    print(f"\nExpected shapes:")
    print(f"   Images: [batch_size, 1, 224, 224] (grayscale, normalized to [0,1])")
    print(f"   Keypoints: [batch_size, 136] (68 points × 2 coords, normalized)")


def save_preprocessing_info():
    """Save preprocessing configuration information."""
    info = f"""
╔═══════════════════════════════════════════════════════════════╗
║           PREPROCESSING CONFIGURATION                         ║
╚═══════════════════════════════════════════════════════════════╝

TRANSFORMATIONS APPLIED:
{'='*60}

Training Pipeline:
  1. Rescale(250)        → Resize images to 250x250
  2. RandomCrop(224)     → Random 224x224 crop (data augmentation)
  3. Normalize()         → Convert to grayscale, normalize to [0,1]
                          → Normalize keypoints: (kpts - 100) / 50
  4. ToTensor()          → Convert to PyTorch tensors

Test Pipeline:
  1. Rescale(224)        → Resize images to 224x224 (no augmentation)
  2. Normalize()         → Same normalization as training
  3. ToTensor()          → Convert to PyTorch tensors

NORMALIZATION FORMULAS:
{'='*60}

Images:
  - Convert to grayscale
  - Normalize: pixel_value / 255.0
  - Range: [0, 1]

Keypoints:
  - Normalize: (keypoint - 100) / 50
  - Range: approximately [-1, 1]
  - To denormalize: keypoint * 50 + 100

BATCH CONFIGURATION:
{'='*60}

  Batch Size: {BATCH_SIZE}
  Image Size: {IMAGE_SIZE}x{IMAGE_SIZE}
  Input Shape: [batch, 1, 224, 224]
  Output Shape: [batch, 136]

DATA AUGMENTATION (Training Only):
{'='*60}

  ✓ RandomCrop: Adds spatial variation
  ✓ Rescale: Ensures consistent input size

{'='*60}
"""
    
    with open('preprocessing_info.txt', 'w') as f:
        f.write(info)
    
    print(info)
    print(f"Saved preprocessing info to 'preprocessing_info.txt'")


def main():
    """Main execution function."""
    print("="*60)
    print("  FACIAL KEYPOINT DETECTION - DATA PREPROCESSING")
    print("="*60)
    
    # Create data loaders
    train_loader, test_loader = create_data_loaders(batch_size=BATCH_SIZE)
    
    if train_loader is None or test_loader is None:
        print(f"\nFailed to create data loaders")
        return
    
    # Inspect batch shapes
    inspect_batch_shapes(train_loader, test_loader)
    
    # Visualize preprocessed samples
    print(f"\n{'─'*60}")
    print(f"VISUALIZING PREPROCESSED DATA")
    print(f"{'─'*60}")
    visualize_batch(train_loader, num_samples=4, save_path='preprocessed_train_samples.png')
    visualize_batch(test_loader, num_samples=4, save_path='preprocessed_test_samples.png')
    
    # Save preprocessing info
    print(f"\n{'─'*60}")
    print(f"SAVING PREPROCESSING CONFIGURATION")
    print(f"{'─'*60}")
    save_preprocessing_info()
    
    print(f"\n{'='*60}")
    print(f"DATA PREPROCESSING COMPLETE!")
    print(f"{'='*60}")
    print(f"\nData loaders are ready for training")
    print(f"Next step: Run 03_train_model.py to train the model")


if __name__ == "__main__":
    main()
