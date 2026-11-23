# Handwritten Digit Classification with PyTorch (MNIST CNN)

A Convolutional Neural Network (CNN) built from scratch in PyTorch to classify handwritten digits (`0`–`9`) from the MNIST dataset, achieving **98.30% accuracy** on 10,000 unseen test images.

## Project Structure

```text
├── dataset.py     # MNIST download, normalization, and train/val/test DataLoaders
├── model.py       # Custom CNN architecture (simple_cnn)
├── train.py       # Training and validation loops + model checkpointing
└── evaluate.py    # Independent test set evaluation
```

## Model Architecture

The network (`simple_cnn`) processes $28 \times 28$ single-channel grayscale images through two convolutional feature-extraction blocks followed by a fully connected classifier head:

1. **Conv Block 1:** `Conv2d(1 → 16, kernel=3, padding=1)` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPool2d(2×2)` *(Output: `16 × 14 × 14`)*
2. **Conv Block 2:** `Conv2d(16 → 32, kernel=3, padding=1)` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPool2d(2×2)` *(Output: `32 × 7 × 7`)*
3. **Classifier Head:** `Flatten` (`1568` features) $\rightarrow$ `Linear(1568 → 128)` $\rightarrow$ `ReLU` $\rightarrow$ `Linear(128 → 10)`

## Training Setup & Results

* **Dataset Split:** 50,000 Training / 10,000 Validation / 10,000 Test images (normalized with `mean=0.1307`, `std=0.3081`)
* **Loss Function:** Cross-Entropy Loss (`nn.CrossEntropyLoss`)
* **Optimizer:** Adam (`lr=0.001`, `batch_size=64`)
* **Epochs:** 5

| Metric | Score |
| :--- | :--- |
| **Final Train Accuracy** | 99.27% (Loss: `0.0233`) |
| **Best Validation Accuracy** | 98.49% (Loss: `0.0479`) |
| **Test Accuracy (10,000 images)** | **98.30%** (Loss: `0.0526`) |

## How to Run

1. **Install dependencies:**
   ```bash
   pip install torch torchvision scikit-learn
   ```

2. **Train the model:**
   ```bash
   python train.py
   ```

3. **Evaluate on the test set:**
   ```bash
   python evaluate.py
   ```