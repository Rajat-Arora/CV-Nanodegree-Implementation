"""
Training script for facial keypoint detection model.

This module handles model training, validation, and checkpointing for
the CNN-based facial keypoint detector.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from pathlib import Path
import json


def train_model(model, train_loader, val_loader, num_epochs=50, 
                learning_rate=0.001, save_dir='saved_models', 
                device='cpu', patience=10):
    """
    Train the facial keypoint detection model.
    
    Args:
        model (nn.Module): The CNN model to train.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        num_epochs (int): Number of epochs to train.
        learning_rate (float): Learning rate for optimizer.
        save_dir (str): Directory to save model checkpoints.
        device (str): Device to train on ('cpu' or 'cuda').
        patience (int): Early stopping patience.
        
    Returns:
        dict: Training history with 'train_loss' and 'val_loss' lists.
    """
    # Create save directory
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Move model to device
    model = model.to(device)
    
    # Training history
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_idx, sample in enumerate(train_loader):
            images = sample['image'].type(torch.FloatTensor).to(device)
            keypoints = sample['keypoints'].type(torch.FloatTensor).to(device)
            
            # Flatten keypoints to (batch_size, 136)
            keypoints = keypoints.view(keypoints.size(0), -1)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, keypoints)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Print progress
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], "
                      f"Batch [{batch_idx+1}/{len(train_loader)}], "
                      f"Loss: {loss.item():.6f}")
        
        # Average training loss
        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for sample in val_loader:
                images = sample['image'].type(torch.FloatTensor).to(device)
                keypoints = sample['keypoints'].type(torch.FloatTensor).to(device)
                keypoints = keypoints.view(keypoints.size(0), -1)
                
                outputs = model(images)
                loss = criterion(outputs, keypoints)
                val_loss += loss.item()
        
        # Average validation loss
        avg_val_loss = val_loss / len(val_loader)
        history['val_loss'].append(avg_val_loss)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] - "
              f"Train Loss: {avg_train_loss:.6f}, "
              f"Val Loss: {avg_val_loss:.6f}")
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            model_path = Path(save_dir) / 'keypoints_model_best.pt'
            torch.save(model.state_dict(), model_path)
            print(f"✓ Model saved: {model_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # Save final model
    final_path = Path(save_dir) / 'keypoints_model_final.pt'
    torch.save(model.state_dict(), final_path)
    
    # Save training history
    history_path = Path(save_dir) / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    return history


def evaluate_model(model, test_loader, device='cpu'):
    """
    Evaluate model on test dataset.
    
    Args:
        model (nn.Module): The trained model.
        test_loader (DataLoader): Test data loader.
        device (str): Device to evaluate on.
        
    Returns:
        float: Average test loss.
    """
    criterion = nn.MSELoss()
    model = model.to(device)
    model.eval()
    
    test_loss = 0.0
    
    with torch.no_grad():
        for sample in test_loader:
            images = sample['image'].type(torch.FloatTensor).to(device)
            keypoints = sample['keypoints'].type(torch.FloatTensor).to(device)
            keypoints = keypoints.view(keypoints.size(0), -1)
            
            outputs = model(images)
            loss = criterion(outputs, keypoints)
            test_loss += loss.item()
    
    avg_test_loss = test_loss / len(test_loader)
    print(f"Average Test Loss: {avg_test_loss:.6f}")
    
    return avg_test_loss
