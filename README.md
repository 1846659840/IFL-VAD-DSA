# Federated Learning for Violence Detection (IFLVADDSA)

This repository implements an **Incremental Federated Learning** approach for Violence Activity Detection using **Decoupled Shapley Aggregation (IFLVADDSA)**. The system is designed to detect violent content in videos through a privacy-preserving federated learning architecture.

---

## Requirements

- **numpy** >= 1.20.0
- **torch** >= 2.0.0
- **torchvision** >= 0.15.0
- **scikit-learn** >= 1.0.0
- **flwr** >= 1.0.0
- **opencv-python** >= 4.5.0

> **Optional:** ImageBind (if not available, ResNet will be used as fallback)

---

## 🌟 Features

- **Multi-modal feature extraction** using ImageBind and Transformer architecture
- **Privacy-preserving federated learning** with the Flower framework
- **Semi-supervised learning** with dynamic pseudo-labeling
- **Adaptive model aggregation** using Shapley value contributions
- **Continual improvement** through incremental learning

---

## 📋 Overview

The system consists of several key components:

### 1. Feature Extraction

The `ImageBindTransformerExtractor` class extracts high-level features from video frames using either:

- Pre-trained ImageBind model (if available)
- ResNet-50 as fallback

Features are then processed through a Transformer architecture to capture temporal relationships.

### 2. Violence Detection Model

The `TransformerViolenceDetector` is a neural network that:

- Processes temporal sequence features
- Uses self-attention mechanisms with positional encoding
- Outputs violence probability for video segments

### 3. Federated Learning System

The system implements a federated learning approach where:

- Multiple clients train models on local private data
- A central server aggregates model updates without accessing raw data
- Knowledge is shared while preserving privacy

### 4. Pseudo-labeling Mechanism

The system employs a two-stage pseudo-labeling approach:

- **Coarse-grained:** Assigns video-level labels based on statistical representations
- **Fine-grained:** Identifies specific violent segments within anomalous videos

### 5. Decoupled Shapley Aggregation

The `IFLVADDSAStrategy` implements a novel aggregation strategy:

- Weights client contributions using Shapley values
- Decouples individual and interactive importance
- Adapts to client data quality and diversity

---

## 🔍 Technical Details

### IFLVADDSA Client

The client performs:

- Local feature extraction
- Statistical analysis for pseudo-labeling
- Local model training
- Adaptive pseudo-label updates based on model predictions

### IFLVADDSA Strategy

The server-side strategy:

- Maintains a global statistical model
- Computes Shapley-based weights for client updates
- Dynamically updates PCA matrix for dimensionality reduction
- Adaptively adjusts pseudo-labeling rules based on client feedback

### Data Processing

- Processes video data in segments of 16 frames
- Extracts features using pre-trained models
- Applies transformer encoding to capture temporal patterns

---

## 🚀 Usage

The system is designed to work with the **XD-Violence dataset，UCF-Crime**. The main simulation can be run with:

```bash
python code.py



