## 🔗 Live Gradio Demo

Check out the full interactive demo here:  
**[Fashion Classifier + Grad-CAM – Live App](https://huggingface.co/spaces/alfred-ogunbayo/fashion-mnist-gradio-demo-app)**

Upload any fashion image (T-shirt, Dress, Sneaker, etc.) and see:
- Predicted class (Top-3)
- Grad-CAM explanation of what part of the image influenced the model

# clothing_recognition_deep_cnn

This project applies deep convolutional neural networks (CNNS) to the Fashion MNIST dataset for clothing item classification.  
It is developed in structured stages to improve generalisation, robustness, and model interpretability using TensorFlow and Hugging Face tools.

---
## Author

Alfred Ogunbayo – MSc AI

Hugging Face: https://huggingface.co/alfred-ogunbayo
GitHub: https://github.com/freddylags

## Dataset

Fashion MNIST:  
60,000 training images and 10,000 test images, each 28×28 grayscale pixels across 10 clothing categories.

---

## Stage Progress

### Stage 1 – Baseline CNN (Completed)
- 2 convolutional layers with Relu + MaxPooling
- Fully connected layers: 64 units + softmax output
- Validation set used during training
- **Test Accuracy:** 90.38%
- **Saved model:** `models/stage1_fashion_cnn.h5`

**Reflection:**  
This baseline CNN performs well, but leaves room for improvement in generalisation and regularisation.  
Next, I will introduce L1 regularisation and cross-validation to reduce potential overfitting and assess performance robustness.

---

## Upcoming Stages

- **Stage 2:** Add L1 Regularisation and K-Fold Cross-Validation
- **Stage 3:** Introduce deeper CNNS with Dropout and BatchNorm
- **Stage 4:** Apply Transfer Learning using pretrained models (EfficientNet, ResNet)
- **Stage 5:** Publish model + metrics to Hugging Face Hub
- **Stage 6:** Visualise with Grad-CAM and build an interactive Gradio demo

---
### Stage 2 – L1 Regularised CNN with K-Fold Validation (Completed)

- Applied L1 weight regularisation to convolutional layers  
- Used 5-fold cross-validation to estimate generalisation performance  
- Tracked accuracy and loss across all folds  
- Automatically saved the best model (based on validation accuracy)

**Metrics:**
- Best Model Fold: Fold 2  
- Best Validation Accuracy: 89.33%  
- Average Accuracy Across Folds: 88.75%  
- Standard Deviation: ±0.0031  
- Saved Model: `models/stage2_fashion_cnn_best_l1.h5`

**Reflection:**  
This stage improved robustness by validating across multiple folds and applying L1 regularisation to reduce overfitting.  
The improvement in validation accuracy and the low standard deviation show stable, consistent performance across data splits.  
Next, I will explore deeper architectures with BatchNorm and Dropout to further improve performance.

---

## Upcoming Stages

- Stage 3: Introduce deeper CNNS with Dropout and BatchNorm  
- Stage 4: Apply Transfer Learning using pretrained models (EfficientNet, ResNet)  
- Stage 5: Publish model and metrics to Hugging Face Hub  
- Stage 6: Visualise with Grad-CAM and build an interactive Gradio demo

---

### Stage 3 – Model Tuning: Deep CNN with Dropout and BatchNorm

Designed a deeper CNN architecture with improved regularisation techniques:
- 3 × Conv2d blocks with Relu activations and Batch Normalisation
- MaxPooling layers for downsampling
- Dropout layers (25% and 50%) to prevent overfitting
- Fully connected layer with 128 units and softmax output

**Training Setup**
- Epochs: 10
- Batch size: 64
- Optimiser: Adam
- Dataset: Fashion MNIST

**Results**
- Test Accuracy: **90.12%**
- Test Loss: **0.2756**
- Saved Model: `models/stage3_fashion_cnn_tuned.h5`
- Training metrics visualised (accuracy/loss plots)

**Reflection**
This stage introduced deeper layers and improved generalisation through Dropout and BatchNorm. Test performance improved vs Stage 2, and the learning process was more stable overall.

---

### Stage 6 – Visualisation and Explainability

- I applied **Grad-CAM** to the final CNN model to interpret the importance of features.
- Generated heat maps are overlaid on the original images.
- Produced 5 visual samples using test data for different clothing categories.
- Visualisation results saved in `gradcam_outputs/` and committed for reproducibility.

**Reflection:**
This final stage provides transparency into the CNN’s decision-making by visualising where the model focuses its attention. These insights help improve trust, explainability, and future model debugging.

---

## Upcoming:
- Deploy the Gradio web app to Hugging Face Spaces
- Enable image upload + live Grad-CAM generation

- ## New Model for Gradio + Grad-CAM (clothing_recognition_deep_cnn_gradio_modelbuilder.h5)

This model was built explicitly to support:

- **Grad-CAM visualisation** via `model.input` and `model.output`
- **Gradio web app integration** for image uploads and heatmap explanation
- Full compatibility with Hugging Face Space deployment

Unlike previous models, which were saved via `Sequential` wrappers, this version was rebuilt using the Functional API to guarantee access to internal layers. This allows clear mapping between input pixels and learned features for interpretability.

### Why this model was added:
> Previous saved models (`stage3_fashion_cnn_tuned.h5`) could not be used with Grad-CAM or deployed via Gradio due to missing `.input` and `.output` references. This new version fixes that.

**File:** `models/clothing_recognition_deep_cnn_gradio_modelbuilder.h5`  
**Notebook:** `notebooks/clothing_recognition_deep_cnn_gradio_modelbuilder.ipynb`

## Requirements

To install dependencies:
```bash
pip install -r requirements.txt