
# clothing_recognition_deep_cnn

This project applies deep convolutional neural networks (CNNs) to the Fashion MNIST dataset for clothing item classification.  
It is developed in structured stages to improve generalisation, robustness, and model interpretability using TensorFlow and Hugging Face tools.

---
## Author

Alfred Ogunbayo – MSc AI

Hugging Face: https://huggingface.co/alfred-ogunbayo

## Dataset

Fashion MNIST:  
60,000 training images and 10,000 test images, each 28×28 grayscale pixels across 10 clothing categories.

---

## Stage Progress

### Stage 1 – Baseline CNN (Completed)
- 2 convolutional layers with ReLU + MaxPooling
- Fully connected layers: 64 units + softmax output
- Validation set used during training
- **Test Accuracy:** 90.38%
- **Saved model:** `models/stage1_fashion_cnn.h5`

**Reflection:**  
This baseline CNN performs well, but leaves room for improvement in generalization and regularization.  
Next, I will introduce L1 regularisation and cross-validation to reduce potential overfitting and assess performance robustness.

---

## Upcoming Stages

- **Stage 2:** Add L1 Regularisation and K-Fold Cross-Validation
- **Stage 3:** Introduce deeper CNNs with Dropout and BatchNorm
- **Stage 4:** Apply Transfer Learning using pretrained models (EfficientNet, ResNet)
- **Stage 5:** Publish model + metrics to Hugging Face Hub
- **Stage 6:** Visualize with Grad-CAM and build an interactive Gradio demo

---

## Requirements

To install dependencies:
```bash
pip install -r requirements.txt