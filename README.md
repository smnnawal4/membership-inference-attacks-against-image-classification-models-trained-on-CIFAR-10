# Membership Inference Attacks against Image Classification Models Trained on CIFAR-10

This project implements **Membership Inference Attacks (MIA)** against image classification models trained on the CIFAR-10 dataset. MIA attacks aim to determine whether a specific data sample was part of a machine learning model's training set, raising important privacy concerns.

## Overview

This project explores three different attack approaches:

1. **Confidence-based attack**: Uses maximum posterior scores from the model
2. **Loss-based attack**: Uses per-sample cross-entropy loss
3. **Shadow-based attack**: Trains an MLP attack model on features extracted from a shadow model

## Project Structure

```
.
├── attacks/              # Attack implementations
│   ├── confidence_based.py
│   └── loss_based.py
├── models/              # Model architectures
│   ├── target_cnn.py
│   ├── shadow_cnn.py
│   ├── attack_model.py
│   └── autoencoder.py
├── utils/               # Utilities
│   ├── attack_features.py
│   ├── training.py
│   └── noise.py
├── notebooks/
│   └── main.ipynb      # Main workflow
└── pyproject.toml
```

## Installation

**Requirements**: Python >= 3.13

```bash
# Using uv (recommended)
uv sync

# Or with pip
pip install torch torchvision scikit-learn tqdm matplotlib numpy
```

## Usage

The complete workflow is available in `notebooks/main.ipynb`. It includes:
- Data loading and preprocessing
- Synthetic data generation using autoencoders
- Target model training
- Confidence and loss-based attacks
- Shadow model training (with synthetic and noisy data)
- Attack model training and evaluation

## Results

**Target Model Performance**:
- Train accuracy: ~98.33%
- Test accuracy: ~67.47%

**Attack Performance**:
- **Confidence-based**: AUC ~0.632
- **Loss-based**: AUC ~0.679
- **Shadow-based (synthetic data)**: Accuracy ~60.05%
- **Shadow-based (noisy data)**: Accuracy ~55.7%

## References

- Shokri, R., et al. (2017). "Membership Inference Attacks Against Machine Learning Models"
- Salem, A., et al. (2019). "ML-Leaks: Model and Data Independent Membership Inference Attacks and Defenses on Machine Learning Models"

## Notes

- Fixed seed (42) for reproducibility
- Models train on CPU by default
- CIFAR-10 dataset is automatically downloaded on first run
