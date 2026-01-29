"""
Script 4: Facial Keypoint Detection Inference

This script performs inference on test images using a trained model:
- Load trained model
- Process test images
- Visualize predictions
- Calculate metrics
- Save results

Usage:
    python 04_inference.py
    
    # Or specify custom model path:
    python 04_inference.py --model saved_models/keypoints_model_best.pt
"""

import os
import sys
from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

# Add parent directory to path to import core modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import our modules
from core.models import Net
from core.data_loader import FacialKeypointsDataset, Rescale, Normalize, ToTensor
from core.inference import FaceDetector, KeypointDetector

# Configuration
DATA_DIR = 'data'
TEST_CSV = os.path.join(DATA_DIR, 'test_frames_keypoints.csv')
SAVE_DIR = '../saved_models'
RESULTS_DIR = 'inference_results'


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run inference on facial keypoint detection')
    parser.add_argument('--model', type=str, default=os.path.join(SAVE_DIR, 'keypoints_model_best.pt'),
                       help='Path to trained model checkpoint')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for inference')
    parser.add_argument('--num_samples', type=int, default=8, help='Number of samples to visualize')
    return parser.parse_args()


def load_model(model_path, device='cpu'):
    """Load trained model from checkpoint."""
    print(f"\n{'='*60}")
    print(f"LOADING MODEL")
    print(f"{'='*60}")
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print(f"\nPlease train the model first using 03_train_model.py")
        return None
    
    print(f"Loading model from: {model_path}")
    
    # Initialize model
    model = Net()
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Load state dict (handle different checkpoint formats)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 'unknown')
        loss = checkpoint.get('loss', 'unknown')
        print(f"Loaded checkpoint from epoch {epoch} with loss {loss:.4f}")
    else:
        model.load_state_dict(checkpoint)
        print(f"Loaded model weights")
    
    model = model.to(device)
    model.eval()
    
    return model


def get_test_loader(batch_size=16):
    """Create test data loader."""
    print(f"\n{'='*60}")
    print(f"PREPARING TEST DATA")
    print(f"{'='*60}")
    
    test_transform = transforms.Compose([
        Rescale((224, 224)),  # Force exact size for batching
        Normalize(),
        ToTensor()
    ])
    
    test_dataset = FacialKeypointsDataset(
        csv_file=TEST_CSV,
        root_dir=os.path.join(DATA_DIR, 'test'),
        transform=test_transform
    )
    
    print(f"Test samples: {len(test_dataset)}")
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"Test batches: {len(test_loader)}")
    
    return test_loader


def calculate_metrics(model, test_loader, device):
    """Calculate evaluation metrics on test set."""
    print(f"\n{'='*60}")
    print(f"CALCULATING METRICS")
    print(f"{'='*60}")
    
    model.eval()
    
    total_loss = 0.0
    all_errors = []
    
    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)
            keypoints_true = batch['keypoints'].to(device)
            
            # Flatten keypoints for loss calculation
            keypoints_true_flat = keypoints_true.view(keypoints_true.size(0), -1)
            
            # Predict
            keypoints_pred = model(images)
            
            # Calculate MSE loss
            loss = torch.nn.functional.mse_loss(keypoints_pred, keypoints_true_flat)
            total_loss += loss.item()
            
            # Calculate per-keypoint Euclidean distance
            kpts_pred = keypoints_pred.cpu().numpy().reshape(-1, 68, 2)
            kpts_true = keypoints_true.cpu().numpy().reshape(-1, 68, 2)
            
            # Denormalize for distance calculation
            kpts_pred = kpts_pred * 50.0 + 100.0
            kpts_true = kpts_true * 50.0 + 100.0
            
            # Calculate Euclidean distance for each keypoint
            distances = np.sqrt(np.sum((kpts_pred - kpts_true) ** 2, axis=2))
            all_errors.extend(distances.flatten())
    
    # Calculate statistics
    avg_loss = total_loss / len(test_loader)
    all_errors = np.array(all_errors)
    
    print(f"\nEvaluation Results:")
    print(f"{'─'*60}")
    print(f"MSE Loss: {avg_loss:.4f}")
    print(f"Mean Error (pixels): {np.mean(all_errors):.2f}")
    print(f"Median Error (pixels): {np.median(all_errors):.2f}")
    print(f"Std Error (pixels): {np.std(all_errors):.2f}")
    print(f"Max Error (pixels): {np.max(all_errors):.2f}")
    print(f"Min Error (pixels): {np.min(all_errors):.2f}")
    print(f"{'─'*60}")
    
    return {
        'mse_loss': avg_loss,
        'mean_error': np.mean(all_errors),
        'median_error': np.median(all_errors),
        'std_error': np.std(all_errors),
        'max_error': np.max(all_errors),
        'min_error': np.min(all_errors)
    }


def visualize_predictions(model, test_loader, device, num_samples=8):
    """Visualize model predictions on test samples."""
    print(f"\n{'='*60}")
    print(f"VISUALIZING PREDICTIONS")
    print(f"{'='*60}")
    
    model.eval()
    
    # Get one batch
    batch = next(iter(test_loader))
    images = batch['image'].to(device)
    keypoints_true = batch['keypoints'].to(device)
    
    # Predict
    with torch.no_grad():
        keypoints_pred = model(images)
    
    # Move to CPU for visualization
    images = images.cpu().numpy()
    keypoints_true = keypoints_true.cpu().numpy()
    keypoints_pred = keypoints_pred.cpu().numpy()
    
    # Select samples
    num_samples = min(num_samples, len(images))
    
    # Create subplot grid
    rows = num_samples
    cols = 2  # Ground truth and prediction
    fig, axes = plt.subplots(rows, cols, figsize=(10, 5 * rows))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for idx in range(num_samples):
        # Get image
        img = images[idx].squeeze()  # Remove channel dim
        img = (img * 255.0).astype(np.uint8)
        
        # Get keypoints
        kpts_true = keypoints_true[idx].reshape(-1, 2)
        kpts_pred = keypoints_pred[idx].reshape(-1, 2)
        
        # Denormalize
        kpts_true = kpts_true * 50.0 + 100.0
        kpts_pred = kpts_pred * 50.0 + 100.0
        
        # Plot ground truth
        ax = axes[idx, 0]
        ax.imshow(img, cmap='gray')
        ax.scatter(kpts_true[:, 0], kpts_true[:, 1], 
                  s=20, c='lime', marker='o', edgecolors='yellow', linewidth=1)
        ax.set_title(f'Sample {idx + 1} - Ground Truth', fontsize=12, fontweight='bold')
        ax.axis('off')
        
        # Plot prediction
        ax = axes[idx, 1]
        ax.imshow(img, cmap='gray')
        ax.scatter(kpts_pred[:, 0], kpts_pred[:, 1], 
                  s=20, c='cyan', marker='o', edgecolors='yellow', linewidth=1)
        ax.set_title(f'Sample {idx + 1} - Prediction', fontsize=12, fontweight='bold')
        ax.axis('off')
    
    plt.tight_layout()
    
    # Save visualization
    os.makedirs(RESULTS_DIR, exist_ok=True)
    save_path = os.path.join(RESULTS_DIR, 'predictions_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved predictions to '{save_path}'")
    plt.show()


def visualize_error_distribution(model, test_loader, device):
    """Visualize error distribution."""
    print(f"\nVisualizing error distribution...")
    
    model.eval()
    all_errors = []
    
    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)
            keypoints_true = batch['keypoints'].to(device)
            
            # Predict
            keypoints_pred = model(images)
            
            # Calculate distances
            kpts_pred = keypoints_pred.cpu().numpy().reshape(-1, 68, 2)
            kpts_true = keypoints_true.cpu().numpy().reshape(-1, 68, 2)
            
            # Denormalize
            kpts_pred = kpts_pred * 50.0 + 100.0
            kpts_true = kpts_true * 50.0 + 100.0
            
            # Calculate Euclidean distance
            distances = np.sqrt(np.sum((kpts_pred - kpts_true) ** 2, axis=2))
            all_errors.extend(distances.flatten())
    
    all_errors = np.array(all_errors)
    
    # Create histogram
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Histogram
    ax1.hist(all_errors, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    ax1.axvline(np.mean(all_errors), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(all_errors):.2f}')
    ax1.axvline(np.median(all_errors), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(all_errors):.2f}')
    ax1.set_xlabel('Error (pixels)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Keypoint Prediction Error Distribution', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Box plot
    ax2.boxplot(all_errors, vert=True)
    ax2.set_ylabel('Error (pixels)', fontsize=12)
    ax2.set_title('Error Box Plot', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    save_path = os.path.join(RESULTS_DIR, 'error_distribution.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved error distribution to '{save_path}'")
    plt.show()


def main():
    """Main execution function."""
    print("="*60)
    print("  FACIAL KEYPOINT DETECTION - INFERENCE")
    print("="*60)
    
    # Parse arguments
    args = parse_args()
    
    # Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load model
    model = load_model(args.model, device)
    if model is None:
        return
    
    # Get test data
    test_loader = get_test_loader(batch_size=args.batch_size)
    
    # Calculate metrics
    metrics = calculate_metrics(model, test_loader, device)
    
    # Visualize predictions
    visualize_predictions(model, test_loader, device, num_samples=args.num_samples)
    
    # Visualize error distribution
    visualize_error_distribution(model, test_loader, device)
    
    print(f"\n{'='*60}")
    print(f"INFERENCE COMPLETE!")
    print(f"{'='*60}")
    print(f"\nResults saved to '{RESULTS_DIR}/' folder")
    print(f"To use the GUI app, run: python app/gui.py")


if __name__ == "__main__":
    main()
