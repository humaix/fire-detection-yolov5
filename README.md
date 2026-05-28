<h1 align="center">🔥 Fire Detection using YOLOv5 & YOLOv9</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Model-YOLOv5%20%7C%20YOLOv9-orange?style=for-the-badge&logo=pytorch" />
  <img src="https://img.shields.io/badge/Framework-Ultralytics-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Interface-Gradio-blueviolet?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Platform-Google%20Colab-yellow?style=for-the-badge&logo=google-colab" />
</p>

<p align="center">
  Real-time fire and flame detection using state-of-the-art YOLO object detection models, 
  complete with a Gradio web interface for image and video inference.
</p>

<p align="center">
  <img src="results/result.gif" width="80%" alt="Fire Detection Demo" />
</p>

---

## 📌 Project Overview

This project implements an **automated fire detection system** using YOLOv5 and YOLOv9 deep learning models. The system can detect fire/flame in:
- **Static images** — bounding box prediction with confidence scores
- **Live video streams / recorded videos** — frame-by-frame fire detection
- **Interactive Gradio demo** — easy-to-use web UI for real-time inference

The model was trained on a filtered version of the [Fire and Gun Dataset](https://www.kaggle.com/datasets/atulyakumar98/fire-and-gun-dataset) from Kaggle, keeping only fire-annotated images.

---

## 📁 Project Structure

```
yolov5-fire-detection/
│
├── fire_model.ipynb       # Gradio inference demo (loads model from Drive or local)
├── train.ipynb            # Full YOLOv5/YOLOv9 training pipeline notebook
├── fire.yaml              # Dataset configuration file (paths + class names)
│
├── model/
│   └── yolov5s_best.pt    # Pretrained YOLOv5s model weights (fire-tuned)
│
├── datasets/
│   └── fire/              # Dataset directory (train/val images + labels)
│
├── results/
│   ├── result.gif         # Demo inference GIF
│   ├── P_curve.png        # Precision curve
│   ├── PR_curve.png       # Precision-Recall curve
│   ├── R_curve.png        # Recall curve
│   ├── F1_curve.png       # F1 score curve
│   ├── results.png        # Training metrics summary
│   ├── val_batch2_labels_*.jpg   # Ground truth validation samples
│   └── val_batch2_pred_*.jpg     # Model predictions on validation samples
│
└── input.mp4              # Sample input video for inference
```

---

## ⚙️ Installation

### 1. Clone this repository
```bash
git clone https://github.com/<your-username>/fire-detection.git
cd fire-detection
```

### 2. Install YOLOv5 dependencies
```bash
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
pip install -r requirements.txt
```

### 3. (Optional) Install YOLOv9 dependencies
```bash
git clone https://github.com/WongKinYiu/yolov9.git
cd yolov9
pip install -r requirements.txt
```

### 4. Install Gradio for the demo interface
```bash
pip install ultralytics gradio
```

---

## 🗂️ Dataset

The model was trained on a custom fire dataset derived from the [Fire and Gun Dataset](https://www.kaggle.com/datasets/atulyakumar98/fire-and-gun-dataset).

**Preprocessing steps applied:**
- Filtered only fire-labeled images (removed gun-only annotations)
- Removed low-resolution images
- Relabeled annotations to a single class: `fire`
- Split into `train/` and `val/` sets

**Dataset configuration (`fire.yaml`):**
```yaml
path: ../datasets/fire/fire
train: train/images
val: val/images
nc: 1
names: ['fire']
```

> ⚠️ The full dataset is **not included** in this repo due to size. Download it from [Kaggle](https://www.kaggle.com/datasets/atulyakumar98/fire-and-gun-dataset) and place it in the `datasets/` folder.

---

## 🏋️ Training

Open `train.ipynb` to run the full training pipeline. Make sure the dataset is placed under `datasets/fire/`.

**YOLOv5 Training Command:**
```bash
python train.py \
  --img 640 \
  --batch 16 \
  --epochs 10 \
  --data ../fire.yaml \
  --weights yolov5s.pt \
  --workers 0
```

**YOLOv9 Training Command:**
```bash
python train_dual.py \
  --workers 4 \
  --device 0 \
  --batch 16 \
  --data ../fire.yaml \
  --img 640 \
  --cfg models/detect/yolov9-c.yaml \
  --weights '' \
  --name yolov9-c \
  --hyp hyp.scratch-high.yaml \
  --min-items 0 \
  --epochs 50 \
  --close-mosaic 15
```

---

## 🔍 Inference

### Using your trained model (YOLOv5):
```bash
python detect.py \
  --source ../input.mp4 \
  --weights runs/train/exp/weights/best.pt \
  --conf 0.2
```

### Using the pretrained model from `model/` folder:
```bash
python detect.py \
  --source ../input.mp4 \
  --weights ../models/yolov5s_best.pt \
  --conf 0.2
```

### YOLOv9 Inference:
```bash
python detect.py \
  --weights runs/train/yolov9-c2/weights/best.pt \
  --source ../input.mp4
```

> 📥 Download the pretrained YOLOv9-c model from [Google Drive](https://drive.google.com/file/d/1nV5C3dbc_Q3CoczHaERTojr78-SFPdMI/view?usp=sharing)

---

## 🖥️ Gradio Web Interface

Open `fire_model.ipynb` in Google Colab or Jupyter. This notebook:

1. **Mounts Google Drive** to load your trained model
2. **Launches a Gradio web UI** with two tabs:
   - 🖼️ **Image Tab** — Upload an image to detect fire
   - 🎥 **Video Tab** — Upload a video for frame-by-frame detection

```python
# Auto-detects environment (Colab or local)
# Loads model from Drive or local path
# Launches Gradio with public share link
demo.launch(share=True)
```

---

## 📊 Results

Training YOLOv5s on the fire dataset for **10 epochs** at 640×640 resolution:

| Metric | Curve |
|:------:|:-----:|
| Precision | ![P Curve](results/P_curve.png) |
| Precision-Recall | ![PR Curve](results/PR_curve.png) |
| Recall | ![R Curve](results/R_curve.png) |
| F1 Score | ![F1 Curve](results/F1_curve.png) |

### Validation Predictions

| Ground Truth | Prediction |
|:---:|:---:|
| ![Label 1](results/val_batch2_labels_1.jpg) | ![Pred 1](results/val_batch2_pred_1.jpg) |
| ![Label 2](results/val_batch2_labels_2.jpg) | ![Pred 2](results/val_batch2_pred_2.jpg) |

### 🔎 Feature Visualization

Visualize internal CNN feature maps using:
```bash
python detect.py \
  --weights runs/train/exp/weights/best.pt \
  --img 640 \
  --conf 0.2 \
  --source ../datasets/fire/val/images/0.jpg \
  --visualize
```

| Input Image | Feature Maps (Stage 23 C3) |
|:-----------:|:--------------------------:|
| ![Input](results/004dec94c5de631f.jpg) | ![Features](results/stage23_C3_features.png) |

---

## 🧠 Key Observations

- ✅ The model performs well on real-world fire scenarios even after just 10 epochs
- ⚠️ Occasionally misclassifies **red emergency lights** (e.g., police car lights) as fire
- 💡 Adding **background/negative samples** (images with no fire) improves false positive rates
- 📖 YOLOv5 authors recommend using **0–10% background images** for best results

---

## 🔗 References

- [YOLOv5 by Ultralytics](https://github.com/ultralytics/yolov5)
- [YOLOv9 by WongKinYiu](https://github.com/WongKinYiu/yolov9)
- [Fire Detection from Images](https://github.com/robmarkcole/fire-detection-from-images)
- [Darknet / AlexeyAB](https://github.com/AlexeyAB/darknet)
- [Fire and Gun Dataset — Kaggle](https://www.kaggle.com/datasets/atulyakumar98/fire-and-gun-dataset)

---

## 👤 Author

**Muhammad Wajiz**  
AI/ML Intern — DevelopersHub Corporation  
📌 *This project is part of an AI/ML Internship program.*

---

<p align="center">
  ⭐ Star this repo if you found it helpful!
</p>
