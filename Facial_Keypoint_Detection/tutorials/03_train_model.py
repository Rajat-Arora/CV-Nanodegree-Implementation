"""
Script 3: Train Facial Keypoint Detection Model

This script trains the CNN model on facial keypoint data with:
- Automatic train/validation split
- Learning rate scheduling
- Early stopping
- Model checkpointing
- Training visualization

Usage:
    python 03_train_model.py
    
    # Or with custom parameters:
    python 03_train_model.py --epochs 100 --batch_size 32 --lr 0.001
"""

import os
import sys
from pathlib import Path
import argparse
import json
import time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

# Add parent directory to path to import core modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import our modules
from core.models import Net
from core.data_loader import FacialKeypointsDataset, Rescale, RandomCrop, Normalize, ToTensor
from core.utils import save_checkpoint, plot_training_history

# Configuration
DATA_DIR = 'data'
TRAINING_CSV = os.path.join(DATA_DIR, 'training_frames_keypoints.csv')
SAVE_DIR = 'saved_models'
IMAGE_SIZE = 224


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Facial Keypoint Detection Model')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation split ratio')
    return parser.parse_args()


def get_data_loaders(batch_size=16, val_split=0.2):
    """Create training and validation data loaders."""
    print(f"\n{'='*60}")
    print(f"PREPARING DATA LOADERS")
    print(f"{'='*60}")
    
    # Create dataset
    train_transform = transforms.Compose([
        Rescale(250),
        RandomCrop(IMAGE_SIZE),
        Normalize(),
        ToTensor()
    ])
    
    full_dataset = FacialKeypointsDataset(
        csv_file=TRAINING_CSV,
        root_dir=os.path.join(DATA_DIR, 'training'),
        transform=train_transform
    )
    
    print(f"Total samples: {len(full_dataset)}")
    
    # Split into train and validation
    val_size = int(val_split * len(full_dataset))
    train_size = len(full_dataset) - val_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Training samples: {train_size}")
    print(f"Validation samples: {val_size}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    return train_loader, val_loader


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    
    for batch_idx, batch in enumerate(train_loader):
        images = batch['image'].to(device)
        keypoints = batch['keypoints'].to(device)

        # Flatten keypoints to [batch, 136] to match model output
        if keypoints.ndim == 3:
            keypoints = keypoints.view(keypoints.size(0), -1)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        
        # Calculate loss
        loss = criterion(outputs, keypoints)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Print progress
        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")
    
    epoch_loss = running_loss / len(train_loader)
    return epoch_loss


def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    
    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device)
            keypoints = batch['keypoints'].to(device)

            # Flatten keypoints to [batch, 136]
            if keypoints.ndim == 3:
                keypoints = keypoints.view(keypoints.size(0), -1)
            
            # Forward pass
            outputs = model(images)
            
            # Calculate loss
            loss = criterion(outputs, keypoints)
            running_loss += loss.item()
    
    val_loss = running_loss / len(val_loader)
    return val_loss


def train_model(model, train_loader, val_loader, epochs=50, lr=0.001, patience=10, device='cpu'):
    """Train the model with early stopping and checkpointing."""
    print(f"\n{'='*60}")
    print(f"TRAINING CONFIGURATION")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Epochs: {epochs}")
    print(f"Learning Rate: {lr}")
    print(f"Early Stopping Patience: {patience}")
    print(f"{'='*60}\n")
    
    # Move model to device
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
    print(scheduler.get_last_lr())
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'lr': []
    }
    
    # Early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Training loop
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        print(f"\n{'─'*60}")
        print(f"Epoch [{epoch + 1}/{epochs}]")
        print(f"{'─'*60}")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        # Print epoch summary
        epoch_time = time.time() - epoch_start
        print(f"\nEpoch [{epoch + 1}/{epochs}] Summary:")
        print(f"   Train Loss: {train_loss:.4f}")
        print(f"   Val Loss: {val_loss:.4f}")
        print(f"   Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"   Time: {epoch_time:.2f}s")
        
        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save best model
            checkpoint_path = os.path.join(SAVE_DIR, 'keypoints_model_best.pt')
            save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path)
            print(f"Saved best model (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epoch(s)")
        
        # Save latest model
        latest_path = os.path.join(SAVE_DIR, 'keypoints_model_latest.pt')
        save_checkpoint(model, optimizer, epoch, val_loss, latest_path)
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break
    
    # Training complete
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE!")
    print(f"{'='*60}")
    print(f"Total Time: {total_time / 60:.2f} minutes")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Final Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
    
    return history


def plot_and_save_history(history, save_path='training_history.png'):
    """Plot and save training history."""
    print(f"\nPlotting training history...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    epochs = range(1, len(history['train_loss']) + 1)
    ax1.plot(epochs, history['train_loss'], 'b-o', label='Train Loss', linewidth=2, markersize=4)
    ax1.plot(epochs, history['val_loss'], 'r-s', label='Val Loss', linewidth=2, markersize=4)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss (MSE)', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot learning rate
    ax2.plot(epochs, history['lr'], 'g-^', linewidth=2, markersize=4)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Learning Rate', fontsize=12)
    ax2.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved training plot to '{save_path}'")
    plt.show()


def save_training_summary(history, args, total_time):
    """Save training summary to JSON."""
    summary = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'epochs_trained': len(history['train_loss']),
        'best_val_loss': min(history['val_loss']),
        'final_train_loss': history['train_loss'][-1],
        'final_val_loss': history['val_loss'][-1],
        'hyperparameters': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.lr,
            'patience': args.patience,
            'val_split': args.val_split
        },
        'training_time_minutes': total_time / 60,
        'history': history
    }
    
    summary_path = os.path.join(SAVE_DIR, 'training_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Saved training summary to '{summary_path}'")


def main():
    """Main execution function."""
    print("="*60)
    print("  FACIAL KEYPOINT DETECTION - MODEL TRAINING")
    print("="*60)
    
    # Parse arguments
    args = parse_args()
    
    # Create save directory
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Get data loaders
    train_loader, val_loader = get_data_loaders(
        batch_size=args.batch_size,
        val_split=args.val_split
    )
    
    # Initialize model
    print(f"\nInitializing model...")
    model = Net()
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model initialized with {num_params:,} parameters")
    
    # Train model
    start_time = time.time()
    history = train_model(
        model, 
        train_loader, 
        val_loader,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        device=device
    )
    total_time = time.time() - start_time
    
    # Plot results
    plot_and_save_history(history)
    
    # Save summary
    save_training_summary(history, args, total_time)
    
    print(f"\n{'='*60}")
    print(f"ALL TRAINING TASKS COMPLETE!")
    print(f"{'='*60}")
    print(f"\nBest model saved to: {os.path.join(SAVE_DIR, 'keypoints_model_best.pt')}")
    print(f"Next step: Run 04_inference.py to test the model")

if __name__ == "__main__":
    main()
