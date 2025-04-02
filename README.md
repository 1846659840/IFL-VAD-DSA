# IFL-VAD-DSA: Federated Learning Violence Detection System

This project implements a violence detection system based on federated learning, utilizing decoupled Shapley aggregation method to enhance joint training effectiveness. The system leverages video feature extraction and Transformer architecture for violence behavior recognition.

## Key Features

- **Federated Learning Architecture**: Uses Flower framework for distributed training
- **Decoupled Shapley Aggregation**: Dynamically adjusts weights based on client contributions
- **Pseudo-label Generation**: Automatically generates and optimizes coarse-grained and fine-grained labels
- **Adaptive Rule Updates**: Dynamically adjusts thresholds based on client feedback

## Project Structure

```
IFL-VAD-DSA/
├── main.py                  # Main program entry
├── requirements.txt         # Dependencies
├── models/                  # Model-related
│   ├── feature_extractor.py # Feature extractor
│   ├── detector.py          # Violence detector
│   └── position_encoding.py # Position encoding
├── data/                    # Data-related
│   ├── dataset.py           # Dataset class
│   └── video_loader.py      # Video loader
├── fl/                      # Federated learning
│   ├── client.py            # Client
│   └── strategy.py          # Strategy
└── utils/                   # Utility functions
    └── helpers.py           # Helper functions
```

## Installing Dependencies

```bash
pip install -r requirements.txt
```

## How to Run

1. Ensure the dataset is prepared (XD-Violence dataset by default)
2. Adjust parameter configurations in `main.py`
3. Run the main program:

```bash
python main.py
```

## Algorithm Flow

1. **Phase I**: Local Pseudo-label Generation
   - Video feature extraction
   - Statistical representation calculation
   - Coarse-grained label assignment
   - Fine-grained label assignment

2. **Phase II**: Local Model Training
   - Train detector based on pseudo-labels
   - Use binary cross-entropy loss
   - Label update and optimization

3. **Phase III**: Aggregation and Global Update
   - Use decoupled Shapley value aggregation
   - Dynamically adjust pseudo-label rules
   - Periodic PCA matrix updates

## Model Architecture

- **Feature Extractor**: Uses ImageBind model (or alternative ResNet)
- **Detector**: Transformer-based sequence classifier
- **Position Encoding**: Standard sinusoidal position encoding

## Citation

If you use this code, please cite:

```
@article{IFL-VAD-DSA,
  title={Interpretable Federated Learning for Video Anomaly Detection with Decoupled Shapley Aggregation},
  author={Your Name},
  year={2023}
}
``` 