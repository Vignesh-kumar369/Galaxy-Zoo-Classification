# Galaxy Zoo Morphology Classification: CNN Pipeline

A PyTorch pipeline for galaxy morphology prediction on the Kaggle *Galaxy Zoo: The Galaxy Challenge* dataset, built as an applied project to gain hands-on experience with image-based deep learning and CNN architecture design.

## Overview

Galaxy Zoo's labels come from a crowdsourced decision tree, where volunteers vote on a galaxy's visual features (smooth vs. spiral, presence of a bar, number of arms, etc.). Each of the 37 target columns is a continuous vote fraction, not a discrete class. This project frames the problem as multi-output regression instead of classification, and trains a CNN end-to-end to predict those 37 vote fractions directly from the galaxy image.

## Dataset

* **Source:** Kaggle *Galaxy Zoo: The Galaxy Challenge*
* **Size:** 61,578 labeled training images (JPEG, distributed as a zip archive)
* **Labels:** 37 continuous values per image, representing hierarchical vote fractions across the Galaxy Zoo decision tree

## Architecture

A 3-block convolutional feature extractor followed by a regression head:

1. Conv2d(3→32, k=3, padding=1) → ReLU → MaxPool2d(2)
2. Conv2d(32→64, k=3, padding=1) → ReLU → MaxPool2d(2)
3. Conv2d(64→128, k=3, padding=1) → ReLU → MaxPool2d(2)
4. Flatten → Linear(128×16×16 → 256) → ReLU → Dropout(0.5) → Linear(256 → 37)

Input images are resized to 128×128. Dropout sits in the classifier head because about 99% of the model's roughly 8.5M parameters are in its first Linear layer, so that is where overfitting risk concentrates.

## Training setup

| Component | Choice |
|---|---|
| Loss | MSE (targets are continuous, not categorical) |
| Optimizer | Adam, lr = 0.001 |
| LR schedule | ReduceLROnPlateau (patience = 3, factor = 0.5) |
| Augmentation (train only) | Random horizontal flip, random vertical flip, random rotation (±180°) |
| Normalization | Per-channel mean/std = 0.5 |
| Epochs | 15 |
| Batch size | 64 |
| Train/validation split | Random 80/20 (49,262 / 12,316 images), random_state = 369 |

Augmentation is applied only to the training split; validation images are only normalized. This is why validation loss sits below training loss throughout training: dropout and augmentation make the training pass harder than evaluation. It is not a sign of a leak or an error.

## Engineering notes

Reading images directly out of the zip archive works single-threaded but breaks under DataLoader's multi-worker setup. The worker processes share the same zip file handle, and concurrent reads corrupt the archive's internal read position. I fixed this by extracting all images to disk before training, so each worker opens its own independent file handle per image.

## Results

Training converges smoothly over 15 epochs. Validation loss decreases and plateaus without rising. Final validation MSE ≈ 0.011 (RMSE about 0.10). This is measured on my own validation split, so it is not comparable to Kaggle leaderboard scores.

![Training vs Validation Loss](<img width="891" height="594" alt="image" src="https://github.com/user-attachments/assets/4d852c06-8531-45f5-b41e-5b46fb1a853b" />
)

## Limitations

* Plain MSE treats the 37 outputs as independent, although the decision tree links them (each question's answers add up to its parent's vote fraction).
* Images are resized whole (424×424 to 128×128) without cropping to the galaxy, so some detail is lost.
* No baseline was run (for example predicting the average vote fractions for every galaxy), and results come from a single run.
* A variant with BatchNorm after each convolution plateaued at validation MSE ≈ 0.027 and was not investigated, so no conclusion is drawn about BatchNorm. The final model has no BatchNorm.
* Results come from a single unseeded run.

## Tools

PyTorch, torchvision, NumPy, PIL, Matplotlib, scikit-learn (train/validation split)

## Scope

This is an applied implementation project built to develop hands-on experience with CNN-based image pipelines (dataset handling, architecture design, regularization, and diagnosing real training issues). It is not a research contribution or an attempt at leaderboard-competitive accuracy.
