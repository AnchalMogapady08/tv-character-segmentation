# 🎬 TV Character Segmentation in Video Frames

A deep learning–based project focused on character-specific semantic segmentation in video frames from TV shows.  
This project implements and compares multiple advanced segmentation architectures to identify and segment characters with high spatial precision.

---

## 📌 Project Overview

The objective of this project is to:

- Extract frames from TV show videos
- Generate weakly-labeled segmentation masks using foundation models
- Train and compare multiple deep learning segmentation architectures
- Evaluate performance using quantitative and qualitative metrics
- Identify the most reliable model for character-level segmentation

---

## 🗂️ Workflow Summary

### 1️⃣ Frame Extraction

- Video frames extracted using **OpenCV**
- Extraction rate: ~1 frame per second
- Organized into character-specific directories
- Structured dataset preparation for segmentation training

---

### 2️⃣ Exploratory Data Analysis (EDA)

Performed to understand dataset characteristics:

- 📊 Frame count per character (bar plot visualization)
- 📐 Image dimension distribution analysis
- 🖼️ Sample frame visualization
- Dataset imbalance inspection

EDA insights guided preprocessing and architecture selection.

---

### 3️⃣ Initial Person Segmentation (Weak Label Generation)

To generate ground truth masks efficiently, a two-stage approach was used:

- **YOLOv8** → Detects bounding boxes for "person" instances  
- **Segment Anything Model (SAM)** → Generates high-quality segmentation masks using bounding box prompts  

Masks were refined using:
- Binarization
- Morphological operations

This created image-mask pairs for supervised training.

---

## 🧠 Advanced Segmentation Models Implemented

All models were implemented using `segmentation_models_pytorch`.

### 🔹 UNet++
- Nested dense skip connections
- Strong feature propagation
- Improved boundary precision

### 🔹 PSPNet (Pyramid Scene Parsing Network)
- Multi-scale pyramid pooling
- Global + local context integration

### 🔹 DINOv2 (Vision Transformer Backbone + FPN)
- Self-supervised transformer encoder
- Rich feature representations
- Integrated within FPN architecture

### 🔹 DeepLabV3
- Atrous (dilated) convolutions
- Atrous Spatial Pyramid Pooling (ASPP)
- Multi-scale contextual understanding

---

## ⚙️ Model Training

- Dataset split: 80% Training / 20% Validation
- Loss Function: Dice Loss
- Optimizer: AdamW
- Evaluation Metric: Dice Score
- Best model checkpoint saved based on validation performance

---

## 📊 Model Performance Comparison

| Model      | Train Loss | Validation Loss | Dice Score |
|------------|------------|----------------|------------|
| DINOv2     | 0.0631     | 0.2843         | **0.7163** |
| UNet++     | 0.1354     | 0.2893         | 0.7128     |
| DeepLabV3  | 0.1007     | 0.2903         | 0.7112     |
| PSPNet     | 0.2194     | 0.3131         | 0.6902     |

---

## 🏆 Key Insights

- **DINOv2** achieved the highest Dice score and best generalization.
- **UNet++** produced the most visually refined segmentation masks with superior boundary accuracy.
- **DeepLabV3** showed consistent and stable performance.
- **PSPNet** demonstrated potential but required additional tuning.

---

## 🔬 Hyperparameter Observations

- Smaller learning rates improved transformer-based DINOv2 stability.
- Dense skip connections in UNet++ improved gradient flow.
- PSPNet required stronger regularization or extended training.
- DeepLabV3 maintained balanced multi-scale feature extraction.

This highlights the importance of **architecture-specific hyperparameter tuning**.

---

## 🖼️ Inference & Visualization

During inference:

- Trained weights loaded per architecture
- Test images preprocessed (resize + normalization)
- Predicted masks generated and visually compared

Qualitative comparisons revealed:
- UNet++ excels in fine detail preservation
- DINOv2 leads numerically
- DeepLabV3 offers smooth outputs


---

## 🛠️ Tech Stack

- Python
- PyTorch
- segmentation_models_pytorch
- OpenCV
- YOLOv8
- Segment Anything Model (SAM)
- Matplotlib
- NumPy

---

## 🎯 Conclusion

While **DINOv2** achieved the highest Dice score (0.7163) and strongest numerical generalization, **UNet++ delivered the most visually accurate and refined segmentations**, making it the most dependable architecture when boundary precision is prioritized.

DeepLabV3 followed as a stable and balanced performer, while PSPNet demonstrated potential with further optimization.

This project demonstrates a complete end-to-end semantic segmentation pipeline — from data preparation to model comparison and inference visualization.

---

## 👩‍💻 Author

Anchal Mogapady  
Aspiring Data Analyst | Machine Learning Enthusiast

---
