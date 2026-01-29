"""
Utility functions for facial keypoint detection project.

Includes helper functions for model management, visualization, and analysis.
"""

import torch
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt


def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """
    Save model checkpoint with optimizer state.
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        epoch: Current epoch number
        loss: Current loss value
        filepath: Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")


def load_checkpoint(model, optimizer, filepath, device='cpu'):
    """
    Load model checkpoint with optimizer state.
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        filepath: Path to checkpoint
        device: Device to load on
        
    Returns:
        tuple: (model, optimizer, epoch, loss)
    """
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    print(f"Checkpoint loaded from epoch {epoch}")
    return model, optimizer, epoch, loss


def count_parameters(model):
    """
    Count total number of trainable parameters in model.
    
    Args:
        model: PyTorch model
        
    Returns:
        int: Total number of parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_summary(model):
    """
    Get summary of model architecture and parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        dict: Model summary statistics
    """
    summary = {
        'total_params': count_parameters(model),
        'model_size_mb': sum(p.numel() * 4 for p in model.parameters()) / (1024**2),
        'layers': len(list(model.parameters())),
    }
    return summary


def plot_training_history(history, save_path=None):
    """
    Plot training and validation loss curves.
    
    Args:
        history: Dictionary with 'train_loss' and 'val_loss' lists
        save_path: Path to save figure (optional)
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = range(1, len(history['train_loss']) + 1)
    ax.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    ax.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss (MSE)', fontsize=12)
    ax.set_title('Model Training History', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Figure saved to {save_path}")
    
    plt.show()
    return fig


def analyze_keypoint_distribution(dataset, num_samples=None):
    """
    Analyze the distribution of keypoints in dataset.
    
    Args:
        dataset: FacialKeypointsDataset instance
        num_samples: Number of samples to analyze (None = all)
        
    Returns:
        dict: Statistics about keypoint distribution
    """
    if num_samples is None:
        num_samples = len(dataset)
    
    all_keypoints = []
    
    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        keypoints = sample['keypoints']
        all_keypoints.append(keypoints)
    
    all_keypoints = np.concatenate(all_keypoints, axis=0)
    
    stats = {
        'mean_x': float(np.mean(all_keypoints[:, 0])),
        'mean_y': float(np.mean(all_keypoints[:, 1])),
        'std_x': float(np.std(all_keypoints[:, 0])),
        'std_y': float(np.std(all_keypoints[:, 1])),
        'min_x': float(np.min(all_keypoints[:, 0])),
        'max_x': float(np.max(all_keypoints[:, 0])),
        'min_y': float(np.min(all_keypoints[:, 1])),
        'max_y': float(np.max(all_keypoints[:, 1])),
    }
    
    return stats


def evaluate_predictions(predictions, ground_truth):
    """
    Evaluate keypoint predictions against ground truth.
    
    Args:
        predictions: Predicted keypoints (N, 68, 2)
        ground_truth: Ground truth keypoints (N, 68, 2)
        
    Returns:
        dict: Evaluation metrics
    """
    # Calculate metrics
    mae = np.mean(np.abs(predictions - ground_truth))
    rmse = np.sqrt(np.mean((predictions - ground_truth) ** 2))
    
    # Per-keypoint distance
    distances = np.sqrt(np.sum((predictions - ground_truth) ** 2, axis=2))
    mean_distance = np.mean(distances)
    
    metrics = {
        'mae': float(mae),
        'rmse': float(rmse),
        'mean_euclidean_distance': float(mean_distance),
        'median_euclidean_distance': float(np.median(distances)),
    }
    
    return metrics


def save_results(results, filepath):
    """
    Save results to JSON file.
    
    Args:
        results: Dictionary of results
        filepath: Path to save file
    """
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {filepath}")


def load_results(filepath):
    """
    Load results from JSON file.
    
    Args:
        filepath: Path to file
        
    Returns:
        dict: Loaded results
    """
    with open(filepath, 'r') as f:
        results = json.load(f)
    return results
