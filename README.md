# Federated Learning for Violence Detection (IFLVADDSA)

This repository implements an **Incremental Federated Learning** approach for Violence Activity Detection using **Decoupled Shapley Aggregation (IFLVADDSA)**. The system is designed to detect violent content in videos through a privacy-preserving federated learning architecture.

---

## 📦 Requirements

```bash
pip install numpy>=1.20.0 torch>=2.0.0 torchvision>=0.15.0 scikit-learn>=1.0.0 flwr>=1.0.0 opencv-python>=4.5.0
https://github.com/adap/flower.git
```
> **Optional:** Install ImageBind for enhanced feature extraction (fallback to ResNet if unavailable).

---

## 🌟 Features

✅ Multi-modal feature extraction using ImageBind and Transformer architecture  
✅ Privacy-preserving federated learning with the Flower framework  
✅ Semi-supervised learning with dynamic pseudo-labeling  
✅ Adaptive model aggregation using Shapley value contributions  
✅ Continual improvement through incremental learning  

---

## 📋 Overview

### 1️⃣ Feature Extraction  
- Uses `ImageBindTransformerExtractor` to extract features from video frames  
- Supports **ImageBind (preferred)** and **ResNet-50 (fallback)**  
- Transformer processes sequential frame relationships  

### 2️⃣ Violence Detection Model  
- Neural network utilizing **self-attention** and **positional encoding**  
- Outputs **violence probability** per video segment  

### 3️⃣ Federated Learning System  
- Clients train locally while a central server **aggregates updates**  
- **No raw data sharing** → **Privacy-preserving training**  

### 4️⃣ Pseudo-labeling Mechanism  
- **Coarse-grained:** Assigns **video-level** labels via statistics  
- **Fine-grained:** Detects **specific violent segments** within anomalous videos  

### 5️⃣ Decoupled Shapley Aggregation  
- Weighs **client contributions dynamically** using **Shapley values**  
- Improves fairness in **model updates**  

---

## 🔍 Technical Details

### 🖥️ Client-Side Processing  
- Extracts **video features**  
- Applies **pseudo-labeling**  
- Trains local models  
- Updates pseudo-labels based on **server feedback**  

### 📡 Server-Side Processing  
- Maintains a **global statistical model**  
- Uses **Shapley-weighted** federated aggregation  
- Adjusts **pseudo-labeling dynamically**  

### 📊 Data Processing Pipeline  
- Processes **video frames in 16-frame segments**  
- Uses **pre-trained models** for feature extraction  
- Applies **transformer encoding** for temporal dependencies  

---

## 🚀 How to Run

The system supports the **XD-Violence dataset，UCF-Crime**. To start training:  

```bash
python code.py
```

### Steps:  
✅ Initializes **clients & server**  
✅ Distributes **data partitions**  
✅ Runs **federated learning simulation (10 rounds)**  
✅ Evaluates **AUC metric**  

---

## 📊 Evaluation Metrics  
- **ROC AUC score** → Measures classification performance  
- **Explicit detection rate** → Identifies violent segments  
- **Privacy metrics** → Ensures federated security  

---

## 🔧 Implementation Details  
- Built with **PyTorch** + **Flower**  
- Supports **GPU acceleration**  
- Adaptive **parameter tuning**  
- **Dynamic pseudo-label thresholds**  

---

## 📄 Citation

If you use this code in your research, please cite:

```bibtex
@article{iflvaddsa2025,
  title={Towards Scalable and Interpretable Federated Learning for Unsupervised Smart City Video Anomaly Detection},
  author={ Wen-Dong Jiang, Graduate Student Member, IEEE, Chih-Yung Chang, Member, IEEE, and Diptendu Sinha Roy, Senior Member, IEEE.},
  journal={sunbmitted to IEEE Trans on Mobile Computing},
  year={2025}
}
```

---

## 📜 License  

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.


