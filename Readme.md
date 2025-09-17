# Chest X-ray Classification using EfficientNet B3

**Author:** Muhammad Huzaifa
**Project Type:** Deep Learning / Computer Vision / Medical Imaging
**Framework:** PyTorch, Streamlit
**Dataset:** Chest X-ray Images (Normal / Pneumonia)

---

## Project Overview

This project implements a **binary classification model** to identify whether a chest X-ray shows a **NORMAL** or **PNEUMONIA** case.

We used **EfficientNet B3 pretrained architecture** and applied **transfer learning** to achieve high accuracy. The model is trained on **Chest X-ray images** with dropout, weight decay, and early stopping for better generalization.

The app includes a **Streamlit interface** for uploading X-ray images and predicting the class in real-time.

---

## Folder Structure

```
Chest_Xray_Project/
│
├── deployment/
│   └── app.py                  # Streamlit demo for inference
├── models/
│   └── model.py                # EfficientNet B3 architecture and classifier head
├── notebooks/
│   └── xray_detection_classifier.ipynb  # Notebook for training & experiments
├── requirements.txt            # Python dependencies
├── README.md                   # This documentation
├── result/
│   ├── loss_curve.png          # Training & validation loss curve
│   └── val_accuracy.png        # Validation accuracy curve
```

> **Note:** The dataset is not included in this repo. The trained weights (`efficientnet_xray_best.pth`) are sufficient for inference.

---

## Dataset

* **Source:** Chest X-ray dataset (Normal / Pneumonia)
* **Structure:**

```
chest_xray/
    train/
        NORMAL/
        PNEUMONIA/
    test/
        NORMAL/
        PNEUMONIA/
```

* **Number of classes:** 2 (Normal, Pneumonia)
* **Training strategy:** Transfer learning with EfficientNet B3

---

## Model Architecture

* Base model: **EfficientNet B3** (pretrained=True, custom classifier)
* Freeze layers: **All except `classifier` layers** → ensures fine-tuning only on last block
* Classifier head:

```python
num_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Linear(num_features, 1536),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(1536, 2)
)
```

* Loss function: `CrossEntropyLoss`
* Optimizer: `Adam` with weight decay (`1e-4`)
* Learning rate scheduler: `ReduceLROnPlateau`
* Early stopping: `patience=3`

---

## Training Details

* **Epochs:** 10
* **Batch size:** 32
* **Device:** GPU / CPU

> Only the classifier layers were trainable, reducing computation while maintaining good feature extraction.

### Epoch-wise Training Log

```
Epoch [1/10] | Train Loss: 0.3497 | Val Loss: 0.4108 | Val Acc: 81.89% | LR: 0.000100
Epoch [2/10] | Train Loss: 0.2216 | Val Loss: 0.3356 | Val Acc: 85.74% | LR: 0.000100
Epoch [3/10] | Train Loss: 0.2147 | Val Loss: 0.3053 | Val Acc: 87.02% | LR: 0.000100
Epoch [4/10] | Train Loss: 0.1959 | Val Loss: 0.3001 | Val Acc: 87.66% | LR: 0.000100
Epoch [5/10] | Train Loss: 0.1907 | Val Loss: 0.2996 | Val Acc: 88.30% | LR: 0.000100
Epoch [6/10] | Train Loss: 0.1882 | Val Loss: 0.2990 | Val Acc: 87.82% | LR: 0.000100
Epoch [7/10] | Train Loss: 0.1837 | Val Loss: 0.2912 | Val Acc: 88.46% | LR: 0.000100
Epoch [8/10] | Train Loss: 0.1823 | Val Loss: 0.2995 | Val Acc: 88.62% | LR: 0.000100
Epoch [9/10] | Train Loss: 0.1751 | Val Loss: 0.3044 | Val Acc: 88.14% | LR: 0.000100
Epoch [10/10] | Train Loss: 0.1826 | Val Loss: 0.3048 | Val Acc: 88.14% | LR: 0.000010
```

---

## Results

* **Validation Accuracy:** 88.14%


### Training Curves

#### Loss Curve

![Loss Curve](result/loss_curve.png)

#### Validation Accuracy Curve

![Validation Accuracy](result/val_accuracy.png)

---

## Observations & Insights

* **Improved generalization:** Validation accuracy increased to 88.14% using EfficientNet B3.
* **Regularization effects:** Dropout (0.5) and weight decay (1e-4) helped reduce overfitting.
* **Data augmentation benefits:** Random flips and rotations increased dataset diversity.
* **Conclusion:** EfficientNet B3 provides a strong balance of accuracy and computational efficiency.

---

## How to Run Streamlit Demo

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the app:

```bash
streamlit run deployment/app.py
```

3. Upload an X-ray image in the browser. The model will predict **NORMAL** or **PNEUMONIA** with a colored result:

* ✅ Green → NORMAL
* ⚠️ Red → PNEUMONIA

---

## Future Improvements

* Fine-tune **more layers** or the entire EfficientNet B3 to further boost accuracy.
* Use **larger dataset** for better generalization.
* Apply **advanced augmentation** (zoom, contrast normalization, random crop).
* Deploy as a **web app** with **Docker** or **Heroku**.

---

## References

1. [EfficientNet Paper](https://arxiv.org/abs/1905.11946)
2. [Chest X-ray Dataset (Kaggle)](https://www.kaggle.com/paultimothymooney/chest-xray-pne)
