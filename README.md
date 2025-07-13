# 🧠 Brain Tumor Detection System  
**(Ensemble Deep Learning Model using ResNet18, EfficientNetB0 & MobileNetV3)**

This project implements an advanced **ensemble-based classifier** to detect brain tumors from MRI scans. The system leverages **transfer learning** from three powerful convolutional neural networks and aggregates their predictions for improved performance.

---

## 🧾 Project Highlights

- ✅ Classifies **MRI brain scans** into:
  - `Glioma`
  - `Meningioma`
  - `Pituitary`
  - `No Tumor`

- ✅ Uses an **Ensemble of 3 pretrained models**:
  - `ResNet18`
  - `EfficientNetB0`
  - `MobileNetV3`

- ✅ Aggregation Method: **Soft Voting** (average of class probabilities)

- ✅ Built using `PyTorch`, `TorchVision`, and `Matplotlib`

---

## 📁 Dataset

- **Source**: [Brain Tumor Dataset on Kaggle](https://www.kaggle.com/datasets/mohammadhossein77/brain-tumors-dataset)  
- **Structure**: The dataset is categorized into 4 folders — one for each tumor type (including "no tumor").  
- **Images**: Preprocessed to size `224x224` and normalized as per ImageNet standards.

---

## 🧠 Architecture Overview

```text
Input MRI Image
       ↓
Preprocessing & Resizing
       ↓
┌────────────┬─────────────┬───────────────┐
│ ResNet18   │ EfficientNetB0 │ MobileNetV3 │
└────────────┴─────────────┴───────────────┘
       ↓
Softmax Probabilities
       ↓
Average (Soft Voting)
       ↓
Final Predicted Class
```
## 📊 Final Evaluation Metrics

> Evaluated on **1311 images** from the test set  
> ✅ **Overall Accuracy**: `88.55%`

| Tumor Class   | 🎯 Precision | 📥 Recall | 📈 F1-Score |
|---------------|-------------|-----------|-------------|
| **Glioma**     | 0.97        | 0.87      | 0.92        |
| **Meningioma** | 0.86        | 0.87      | 0.86        |
| **No Tumor**   | 0.96        | 0.97      | 0.97        |
| **Pituitary**  | 0.91        | 0.99      | 0.95        |

## 🧪 Example Prediction (Single Image)

```text
🧠 Prediction Results:
👉 Predicted Class: pituitary

📊 Class Probabilities:
glioma: 0.0042  
meningioma: 0.0171  
no_tumor: 0.0003  
pituitary: 0.9784
```
## 🚀 Running the Code

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/brain-tumor-detection.git
cd brain-tumor-detection
```
### 2. Install Dependencies

```bash
pip install -r requirements.txt

```
3. Launch the Notebook

```bash
Open FinalEnsemble.ipynb in Jupyter or VS Code, and run all the cells.
```
4. Choose Prediction Mode

```bash
Enter 'single' → to test a single image
Enter 'batch'  → to evaluate a test directory

```

## 📈 Training History
The training and validation accuracy and loss curves were visualized to monitor the model's performance and check for overfitting. These plots help ensure the model generalizes well to unseen data.

## ✅ Conclusion
This deep learning system classifies brain MRI scans with an accuracy of **88.55%** by leveraging an ensemble of state-of-the-art CNN architectures. It shows strong potential to assist radiologists and diagnostic systems by delivering fast and accurate predictions, thereby improving clinical decision-making.

