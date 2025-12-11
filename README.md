You can copy–paste directly into **README.md**.

---

# 🚀 **Dual-Stage Kalman Framework for Water-Body Segmentation**

### *Spatial Denoising + Temporal Prediction using Deep Learning + Kalman Filtering*

---

## 📌 **Overview**

This repository contains the implementation of a **Dual-Stage Kalman Framework** for accurate and lightweight water-body extraction from satellite imagery.
The system integrates:

### **🔹 Module 1 — LKF-SegNet (Spatial Segmentation Engine)**

* U-Net decoder
* MobileNetV2 lightweight encoder
* **Spatial Learnable Kalman Filter (LKF)** at the bottleneck
* **Edge-Weighted Loss** for sharp water–land boundaries

### **🔹 Module 2 — Temporal Kalman Filter (TKF)**

* Applied on sequences of probability maps
* Handles synthetic video generated via sliding-window
* Produces smoothed and **one-step-ahead predicted masks**

This hybrid deep-learning + signal-processing approach improves **spatial coherence**, **temporal stability**, and **prediction capability** while remaining computationally efficient for edge devices (drones, IoT, etc.).

---

## 📂 **Repository Structure**

```
├── model/
│   ├── lkf_segnet.py            # Full LKF-SegNet implementation
│   ├── kalman_filter_spatial.py # Learnable Spatial Kalman Filter module
│   ├── kalman_filter_temporal.py# Temporal Kalman Filter implementation
│
├── training/
│   ├── train_lkf_segnet.py      # Training loop with edge-weighted loss
│   ├── loss_edge_weighted.py    # Canny + distance-transform loss
│
├── synthetic_video/
│   ├── generate_synthetic_video.py
│   ├── run_temporal_filter.py
│
├── utils/
│   ├── dataset_loader.py
│   ├── augmentation.py
│   ├── metrics.py
│
├── results/
│   ├── mae_plot.png
│   ├── iou_distribution.png
│   ├── train_vs_test_metrics.png
│   ├── qualitative_segmentation.png
│   ├── temporal_outputs.png
│
├── README.md
└── requirements.txt
```

---

## 🧠 **Model Architecture**

### 🔷 **LKF-SegNet Overview**

*(Insert your image here in GitHub)*

```
![LKF-SegNet Architecture](images/lkf_segnet_architecture.png)
```

A U-Net–style encoder–decoder network enhanced with a **Spatial Learnable Kalman Filter** to stabilize and denoise feature maps.

---

### 🔷 **Row-Wise Learnable Kalman Filtering**

```
![Row-wise LKF](images/lkf_internal_flow.png)
```

Each row of the bottleneck feature map is treated as a pseudo-temporal sequence, enabling recursive prediction–correction at the feature level.

---

## 📦 **Dataset Preparation**

* Images resized to **256×256**
* Binary masks (1 = water, 0 = non-water)
* Augmentations used:

  * Horizontal/vertical flips
  * Color normalization
  * Random cropping

### **Synthetic Video Generation**

A sliding window is used to create an ordered sequence of overlapping patches from a single large satellite image:

```
python generate_synthetic_video.py --image path/to/image.png
```

---

## 🔧 **Installation**

```bash
git clone https://github.com/yourusername/dual-stage-kalman-water-body-segmentation
cd dual-stage-kalman-water-body-segmentation
pip install -r requirements.txt
```

---

## 🏋️ **Training LKF-SegNet**

```bash
python train_lkf_segnet.py \
    --dataset path/to/dataset \
    --epochs 50 \
    --batch_size 8
```

Training uses:

* MobileNetV2 encoder
* Edge-Weighted Loss
* Spatial Learnable Kalman Filter

---

## 🎬 **Run Temporal Kalman Filtering**

```bash
python run_temporal_filter.py --input_folder synthetic_video/
```

Outputs:
✔ Smoothed masks
✔ One-step-ahead predictions
✔ MAE plots

---

## 📊 **Results**

### **1️⃣ MAE Over Time (First 50 Frames)**

Shows temporal behavior of:

* Observation ( z_t )
* Filtered state ( x_t )
* Predicted state ( x_{t|t-1} )

The filter reduces flicker and stabilizes predictions.

```
![MAE Plot](results/mae_plot.png)
```

---

### **2️⃣ Temporal Filtering Example**

For one frame:

* RGB patch
* Ground Truth
* Observation
* Filtered output
* Predicted output

```
![Temporal Outputs](results/temporal_outputs.png)
```

The filter smooths noise and creates stable predictions.

---

### **3️⃣ Train vs Test Metrics**

```
![Train vs Test](results/train_vs_test_metrics.png)
```

High generalization:

* Accuracy ≈ 89%
* IoU ≈ 75%
* Precision ≈ 89%
* Recall ≈ 83%

---

### **4️⃣ IoU Distribution**

```
![IoU Distribution](results/iou_distribution.png)
```

Most IoUs lie within **0.70–0.90**, indicating consistent segmentation performance.

---

### **5️⃣ Qualitative Segmentation**

```
![Qualitative](results/qualitative_segmentation.png)
```

The model preserves boundary sharpness and captures water regions accurately.

---

## 🚀 **Key Contributions**

✔ A **lightweight** MobileNetV2-based segmentation model
✔ Integration of **Spatial Learnable Kalman Filtering**
✔ **Edge-Weighted Loss** for crisp water boundaries
✔ **Synthetic temporal dataset** to evaluate temporal filtering
✔ Temporal Kalman Filter enabling **prediction + smoothing**
✔ Detailed experiments demonstrating improvement over raw CNN outputs

---

## 📚 **Citation**

If you use this work, please cite:

```
@article{your_kalman_2025,
  title={Dual-Stage Kalman Framework for Spatial and Temporal Water-Body Segmentation},
  author={Your Name},
  year={2025}
}
```
---

## 🤝 **Contributions**

Pull requests are welcome!
If you find issues, open an issue with screenshots and logs.
