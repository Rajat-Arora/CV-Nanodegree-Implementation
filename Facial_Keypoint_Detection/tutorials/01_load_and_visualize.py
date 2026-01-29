"""
Script 1: Load and Visualize Facial Keypoint Data

This script loads the facial keypoint dataset and visualizes sample images
with their corresponding keypoints.

Based on Udacity's "1. Load and Visualize Data.ipynb" notebook.

Usage:
    python 01_load_and_visualize.py
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from pathlib import Path

# Configuration
DATA_DIR = 'data'
TRAINING_CSV = os.path.join(DATA_DIR, 'training_frames_keypoints.csv')
TEST_CSV = os.path.join(DATA_DIR, 'test_frames_keypoints.csv')
SAMPLE_COUNT = 3  # Number of samples to visualize


def load_data(csv_path):
    """Load facial keypoint data from CSV file."""
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        print(f"\nExpected data structure:")
        print(f"   {DATA_DIR}/")
        print(f"   ├── training_frames_keypoints.csv")
        print(f"   ├── test_frames_keypoints.csv")
        print(f"   ├── training/ (folder with images)")
        print(f"   └── test/ (folder with images)")
        print(f"\nDownload the YouTube Faces Dataset:")
        print(f"   https://s3.amazonaws.com/video.udacity-data.com/topher/2018/May/5aea1b91_train-test-data/train-test-data.zip")
        print(f"\n   Then extract to the data/ folder.")
        return None
    
    print(f"Loading data from {csv_path}...")
    key_pts_frame = pd.read_csv(csv_path)
    print(f"Loaded {len(key_pts_frame)} samples")
    return key_pts_frame


def show_keypoints(image, key_pts):
    """Show image with keypoints - matches notebook function."""
    plt.imshow(image)
    plt.scatter(key_pts[:, 0], key_pts[:, 1], s=20, marker='.', c='m')


def display_single_sample(key_pts_frame, sample_idx=0):
    """Display a single sample with stats - matches notebook approach."""
    print(f"\n{'─'*60}")
    print(f"SAMPLE {sample_idx} DETAILS")
    print(f"{'─'*60}")
    
    # Get image name and keypoints
    image_name = key_pts_frame.iloc[sample_idx, 0]
    key_pts = key_pts_frame.iloc[sample_idx, 1:].values
    key_pts = key_pts.astype('float').reshape(-1, 2)
    
    print(f'Image name: {image_name}')
    print(f'Landmarks shape: {key_pts.shape}')
    print(f'First 4 key pts:\n{key_pts[:4]}')
    
    return image_name, key_pts


def visualize_samples(key_pts_frame, data_type='training', num_samples=3):
    """Visualize multiple sample images with keypoints - matches notebook style."""
    print(f"\nVisualizing {num_samples} {data_type} samples...")
    
    # Get image directory
    img_dir = os.path.join(DATA_DIR, data_type)
    
    if not os.path.exists(img_dir):
        print(f"Error: Image directory not found at {img_dir}")
        print(f"Make sure to extract the dataset to: {DATA_DIR}/")
        return
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 10))
    
    for i in range(num_samples):
        # Randomly select a sample
        rand_i = np.random.randint(0, len(key_pts_frame))
        
        # Get image name and keypoints
        image_name = key_pts_frame.iloc[rand_i, 0]
        key_pts = key_pts_frame.iloc[rand_i, 1:].values
        key_pts = key_pts.astype('float').reshape(-1, 2)
        
        # Load image using matplotlib's imread
        img_path = os.path.join(img_dir, image_name)
        
        if not os.path.exists(img_path):
            print(f"⚠️  Image not found: {img_path}")
            continue
        
        image = mpimg.imread(img_path)
        
        # If image has alpha channel, remove it
        if len(image.shape) == 3 and image.shape[2] == 4:
            image = image[:, :, 0:3]
        
        # Print shape info
        print(f"{i}: Image shape: {image.shape}, Keypoints shape: {key_pts.shape}")
        
        # Create subplot
        ax = plt.subplot(1, num_samples, i + 1)
        ax.set_title(f'Sample #{rand_i}')
        
        # Display image with keypoints using the show_keypoints function
        show_keypoints(image, key_pts)
    
    plt.tight_layout()
    save_path = f'{data_type}_samples_visualization.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to '{save_path}'")
    plt.show()


def analyze_dataset(key_pts_frame):
    """Analyze and print dataset statistics - matches notebook approach."""
    print(f"\nDataset Statistics:")
    print(f"{'='*60}")
    print(f"Number of images: {key_pts_frame.shape[0]}")
    print(f"Number of features (including image name): {key_pts_frame.shape[1]}")
    print(f"Number of keypoints: {(key_pts_frame.shape[1] - 1) // 2}")  # 68 keypoints
    
    # Check for missing values
    missing_count = key_pts_frame.isnull().sum().sum()
    if missing_count > 0:
        print(f"Warning: Missing values: {missing_count}")
    else:
        print(f"No missing values")
    
    # Keypoint statistics
    keypoint_cols = key_pts_frame.iloc[:, 1:]  # Exclude image name column
    print(f"\nKeypoint Value Ranges:")
    print(f"   Min value: {keypoint_cols.min().min():.2f}")
    print(f"   Max value: {keypoint_cols.max().max():.2f}")
    print(f"   Mean value: {keypoint_cols.mean().mean():.2f}")
    print(f"   Std value: {keypoint_cols.std().mean():.2f}")
    print(f"{'='*60}\n")


def main():
    """Main execution function - follows notebook workflow."""
    print("="*60)
    print("  FACIAL KEYPOINT DETECTION - DATA LOADING & VISUALIZATION")
    print("  Based on Udacity's Notebook 1")
    print("="*60)
    
    # Check if data directory exists
    if not os.path.exists(DATA_DIR):
        print(f"\nError: Data directory '{DATA_DIR}' not found!")
        print(f"\nCreating data directory structure...")
        os.makedirs(DATA_DIR, exist_ok=True)
        os.makedirs(os.path.join(DATA_DIR, 'training'), exist_ok=True)
        os.makedirs(os.path.join(DATA_DIR, 'test'), exist_ok=True)
        print(f"Created directory structure")
        print(f"\nDownload the YouTube Faces Dataset:")
        print(f"   URL: https://s3.amazonaws.com/video.udacity-data.com/topher/2018/May/5aea1b91_train-test-data/train-test-data.zip")
        print(f"\n   Extract the contents to the '{DATA_DIR}/' folder")
        print(f"   You should have:")
        print(f"   - {DATA_DIR}/training_frames_keypoints.csv")
        print(f"   - {DATA_DIR}/test_frames_keypoints.csv")
        print(f"   - {DATA_DIR}/training/ (folder with images)")
        print(f"   - {DATA_DIR}/test/ (folder with images)")
        return
    
    # Load training data
    print(f"\n{'─'*60}")
    print(f"STEP 1: LOADING TRAINING DATA")
    print(f"{'─'*60}")
    key_pts_frame = load_data(TRAINING_CSV)
    
    if key_pts_frame is None:
        return
    
    # Display dataset statistics
    analyze_dataset(key_pts_frame)
    
    # Display a single sample with details
    image_name, key_pts = display_single_sample(key_pts_frame, sample_idx=1)
    
    # Visualize multiple samples
    print(f"\n{'─'*60}")
    print(f"STEP 2: VISUALIZING TRAINING SAMPLES")
    print(f"{'─'*60}")
    visualize_samples(key_pts_frame, data_type='training', num_samples=SAMPLE_COUNT)
    
    # Load and visualize test data
    print(f"\n{'─'*60}")
    print(f"STEP 3: LOADING TEST DATA")
    print(f"{'─'*60}")
    test_key_pts_frame = load_data(TEST_CSV)
    
    if test_key_pts_frame is not None:
        analyze_dataset(test_key_pts_frame)
        
        print(f"\n{'─'*60}")
        print(f"STEP 4: VISUALIZING TEST SAMPLES")
        print(f"{'─'*60}")
        visualize_samples(test_key_pts_frame, data_type='test', num_samples=SAMPLE_COUNT)
    
    print(f"\n{'='*60}")
    print(f"Data loading and visualization complete!")
    print(f"{'='*60}")
    print(f"\nKey Observations:")
    print(f"   • Dataset contains {len(key_pts_frame)} training images")
    if test_key_pts_frame is not None:
        print(f"   • Dataset contains {len(test_key_pts_frame)} test images")
    print(f"   • Each image has 68 facial keypoints (136 values)")
    print(f"   • Images vary in size - will need standardization")
    print(f"\nNext step: Run 02_preprocess_data.py to prepare data for training")


if __name__ == "__main__":
    main()
