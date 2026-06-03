# UCF-Crime Video Anomaly Detection

An end-to-end deep learning pipeline for detecting anomalies in surveillance videos using the UCF-Crime dataset.

## 📊 Results

| Metric | Score |
|--------|-------|
| Top-1 Accuracy | 49.00% |
| Top-5 Accuracy | 73.05% |

## 🏗️ Architecture

- **CNN Backbone**: MobileNetV2 (frozen layers)
- **Temporal Model**: LSTM (128 units)
- **Sequence Length**: 16 frames
- **Input Size**: 224×224×3

## 📁 Dataset

UCF-Crime dataset with 13 anomaly classes:
- Abuse, Arrest, Arson, Assault, Burglary, Explosion, Fighting
- NormalVideos, RoadAccidents, Robbery, Shooting, Shoplifting, Stealing, Vandalism

## 🚀 Features

- Checkpoint-based dataset processing (resume from interruption)
- Video-level train/val split (prevents data leakage)
- Multi-GPU support (DataParallel)
- Top-1 and Top-5 accuracy metrics
- Real-time inference support

## 🛠️ Installation

```bash
pip install torch torchvision opencv-python numpy pandas scikit-learn tqdm
