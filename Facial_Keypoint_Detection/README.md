# Facial Keypoint Detection

A deep learning system for detecting 68 facial keypoints using Convolutional Neural Networks. Includes a real time GUI application with webcam support.

[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12.1-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## Table of Contents

- [Overview](#overview)
- [Results](#results)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)

---

## Overview

This project implements a deep learning model to detect **68 facial keypoints** (facial landmarks) using Convolutional Neural Networks (CNNs). The model predicts precise locations of facial features like eyes, eyebrows, nose, mouth, and jawline, enabling applications in:

- **Face Filters & AR Effects** - Snapchat style filters
- **Emotion Recognition** - Facial expression analysis
- **Face Alignment** - Preprocessing for face recognition
- **Pose Estimation** - Head orientation detection
- **Facial Analysis** - Research and diagnostics

**Dataset**: YouTube Faces Dataset (3,462 training + 2,308 test images)

---

## Results


### Model Predictions

<p align="center">
  <img src="../media/inference_results/predictions_comparison.png" alt="Predictions Comparison" width="800"/>
  <br>
  <em>Ground truth vs. model predictions on test set</em>
</p>

### Error Analysis

<p align="center">
  <img src="../media/inference_results/error_distribution.png" alt="Error Distribution" width="800"/>
  <br>
  <em>Distribution of prediction errors across all keypoints</em>
</p>

---

## Quick Start

### Prerequisites
- **Python 3.8+** (3.8 recommended)
- **4GB RAM** minimum (8GB+ recommended)
- **Webcam** (for GUI app)
- **500MB disk space**


## Installation

### Using pip

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

---

## Usage

### Complete Pipeline (Recommended for First-Time Users)

Follow these scripts in order for a complete workflow. All tutorial scripts are located in the `tutorials/` folder.

**Important**: The tutorial scripts expect the data to be in the `tutorials/data/` directory. Make sure your dataset is placed there before running the scripts.

#### Step 1: Load and Visualize Data
```bash
python3 tutorials/01_load_and_visualize.py
```
**What it does:**
- Loads facial keypoint data from CSV files (from `tutorials/data/`)
- Displays dataset statistics
- Visualizes sample images with keypoints
- Checks data integrity

**Output:** `visualization_samples.png` with annotated faces

#### Step 2: Preprocess Data
```bash
python3 tutorials/02_preprocess_data.py
```
**What it does:**
- Creates PyTorch data loaders
- Applies transformations (rescale, crop, normalize)
- Validates preprocessed data
- Shows before/after preprocessing

**Output:** `preprocessed_samples.png`, `preprocessing_info.txt`

#### Step 3: Train Model
```bash
python3 tutorials/03_train_model.py --epochs 50 --batch_size 16 --lr 0.001
```
**What it does:**
- Trains the CNN model
- Applies early stopping
- Saves checkpoints
- Plots training curves

**Output:** 
- `../saved_models/keypoints_model_best.pt` (best model - saved in parent directory)
- `training_history.png` (loss curves)
- `training_summary.json` (metrics)

**Training Options:**
```bash
# Quick test (5 epochs)
python3 tutorials/03_train_model.py --epochs 5

# Full training with custom parameters
python tutorials/03_train_model.py --epochs 100 --batch_size 32 --lr 0.0001 --patience 15
```

#### Step 4: Run Inference
```bash
python tutorials/04_inference.py --model ../saved_models/keypoints_model_best.pt
```
**What it does:**
- Loads trained model
- Evaluates on test set
- Calculates metrics (MSE, mean error, etc.)
- Visualizes predictions vs ground truth
- Shows error distribution

**Output:**
- `inference_results/predictions_comparison.png`
- `inference_results/error_distribution.png`
- Detailed metrics in console

---

### Real-Time GUI Application

Launch the GUI for real time facial keypoint detection:

```bash

# Launch GUI
python app/gui.py
```

**GUI Features:**
1. **Camera Selection** - Choose from available webcams
2. **Start/Stop Detection** - Control processing
3. **Adjustable Parameters**:
   - Scale Factor (1.0-1.5): Detection sensitivity
   - Min Neighbors (1-10): Detection specificity
   - Show Face Box: Toggle bounding boxes
   - Use GPU: Enable CUDA acceleration
4. **Real-Time Stats**: FPS, face count, keypoints detected

**Controls:**
- `Start Detection` - Begin processing
- `Stop Detection` - Pause processing
- Adjust sliders for fine-tuning
- Select different cameras from dropdown

**Performance:**
- **GPU (CUDA)**: 10 FPS

---