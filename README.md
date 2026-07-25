# Galaxy Zoo Morphology Classification — CNN Pipeline

A PyTorch pipeline for galaxy morphology classification on the [Kaggle Galaxy Zoo: The Galaxy Challenge](https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge) dataset, built as an applied project to gain hands-on experience with image-based deep learning and CNN architecture design.

## Overview

Galaxy Zoo's labels come from a crowdsourced decision tree, where volunteers vote on a galaxy's visual features (smooth vs. spiral, presence of a bar, number of arms, etc.). Each of the 37 target columns is a **continuous vote fraction**, not a discrete class. This project frames the problem as multi-output regression rather than classification, and trains a CNN end-to-end to predict those 37 vote fractions directly from the galaxy image.

## Dataset

- **Source:** Kaggle Galaxy Zoo: The Galaxy Challenge
- **Size:** 61,578 labeled training images (JPEG, distributed as a zip archive)
- **Labels:** 37 continuous values per image, representing hierarchical vote fractions across the Galaxy Zoo decision tree

## Architecture

A 3-block convolutional feature extractor followed by a regression head:

```
Conv2d(3→32, k=3) → ReLU → MaxPool2d(2)
Conv2d(32→64, k=3) → ReLU → MaxPool2d(2)
Conv2d(64→128, k=3) → ReLU → MaxPool2d(2)
Flatten → Linear(128×16×16 → 256) → ReLU → Dropout(0.5) → Linear(256 → 37)
```

Input images are resized to 128×128. Dropout is placed in the classifier head, where the fully-connected layer's parameter count is far higher than in the convolutional blocks and overfitting risk concentrates.

## Training setup

| Component | Choice |
|---|---|
| Loss | MSE (targets are continuous, not categorical) |
| Optimizer | Adam, lr = 0.001 |
| LR schedule | ReduceLROnPlateau (patience=3, factor=0.5) |
| Augmentation (train only) | Random horizontal flip, random vertical flip, random rotation (±180°) |
| Normalization | Per-channel mean/std = 0.5 |
| Epochs | 15 |
| Batch size | 64 |

Augmentation is applied only to the training split; validation images are only normalized, which is why validation loss tracks *below* training loss throughout training — dropout and augmentation both handicap the training forward pass relative to evaluation, not a sign of a leak or an error.

## Engineering notes

Reading images directly out of the zip archive works fine single-threaded, but breaks under `DataLoader`'s multi-worker setup: each worker process shares the same zip file handle, and concurrent reads corrupt the archive's internal read position. Fixed by extracting all images to disk before training, so each worker opens its own independent file handle per image.

## Results

Training converges smoothly over 15 epochs with no evidence of overfitting — validation loss decreases and plateaus without diverging upward. Final validation MSE ≈ **0.011**.

<img width="998" height="496" alt="image" src="https://github.com/user-attachments/assets/e0fbd80e-0b0c-444a-ba7e-96074617b051" />

## Tools

PyTorch, torchvision, NumPy, PIL, Matplotlib, scikit-learn (train/validation split)

## Scope

This is an applied implementation project built to develop hands-on experience with CNN-based image pipelines — dataset handling, architecture design, regularization, and diagnosing real training issues — not a research contribution or an attempt at leaderboard-competitive accuracy.
