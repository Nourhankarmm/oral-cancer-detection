# 🩺 Oral Cancer Detection (Lips and Tongue) using CNNs & Vision Transformers

A deep learning project to classify oral cavity images (lips and tongue) into cancerous and non-cancerous classes using advanced image classification techniques.

![cover](assets/cover.jpg)

---

## 🚀 Project Overview

Early detection of oral cancer can save lives. This project aims to develop a computer vision model that classifies oral images (tongue and lips) as:
- ✅ Non-Cancerous
- ❌ Cancerous

The model leverages both Convolutional Neural Networks (CNNs) and Transformer-based architectures to improve diagnostic accuracy and reduce manual clinical workload.

---

## 🧠 Models Used

| Model           | Type               | Description                         |
|----------------|--------------------|-------------------------------------|
| CNN (Custom)   | Convolutional NN   | Lightweight model for classification |
| ResNet50       | Transfer Learning  | Pretrained on ImageNet              |
| ViT (Vision Transformer) | Transformer-based | High-performance deep model         |

---

## 📁 Dataset Structure

<pre>
dataset/
├── cancer/           # Images showing signs of oral cancer
└── non_cancer/       # Healthy images of lips and tongue
</pre>

- **Source**: [Kaggle - Oral Cancer Image Dataset](https://www.kaggle.com/datasets/tahmidmir/oral-cancer)

---

## 🏗️ Project Structure

<pre>
├── models/
│   ├── cnn_model.py
│   ├── resnet_model.py
│   └── vit_model.py
│
├── dataset/
│   ├── cancer/
│   └── non_cancer/
│
├── notebooks/
│   └── oral_cancer_classification.ipynb
│
├── assets/
│   ├── cover.jpg
│   ├── img_cancer.jpg
│   └── img_healthy.jpg
│
├── requirements.txt
└── README.md
</pre>

---

## 🧪 Sample Predictions

| Sample Image           | Model Prediction     |
|------------------------|----------------------|
| ![](assets/img_cancer.jpg) | Cancer Detected ❌     |
| ![](assets/img_healthy.jpg) | Healthy ✅            |

---

## 📊 Evaluation Metrics

- Accuracy
- Precision
- Recall (Sensitivity)
- Specificity
- F1 Score
- Confusion Matrix

---

## ⚙️ Installation

git clone https://github.com/Nourhankarmm/oral-cancer-classification.git
cd oral-cancer-classification
pip install -r requirements.txt

## 🛠️ Tech Stack
Python
TensorFlow / PyTorch
Transfer Learning (ResNet50)
Vision Transformers (ViT via HuggingFace)
OpenCV
NumPy, Matplotlib, Scikit-learn
