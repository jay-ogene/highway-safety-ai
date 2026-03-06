# 🚗 AI Highway Safety & Incident Detection Platform

> A real-time AI system for detecting high-risk highway behaviors and
> generating safety analytics through a scalable cloud infrastructure.

[DEMO GIF GOES HERE — added Week 6]

## 🎯 What This Does
Real-time detection of dangerous highway behaviors using a custom-trained
YOLOv8 model + multi-object tracking + predictive near-miss detection.

## ⚡ Key Results
| Metric              | Value         |
| ------------------- | ------------- |
| mAP@0.5             | [Week 2]      |
| Inference Speed     | [Week 2] FPS  |
| Alert Latency       | [Week 5] ms   |
| Near-miss Precision | [Week 3]      |

## 🏗️ Architecture
[Architecture diagram — added Week 6]

## 📦 Tech Stack
- Detection: YOLOv8m (Ultralytics)
- Tracking: ByteTrack
- Near-Miss: Kalman filter + TTC estimation
- Cloud: GCP (Vertex AI, Pub/Sub, BigQuery, Firestore, Cloud Run)
- Experiment Tracking: Weights & Biases

## 🚀 Quick Start
[Instructions — added Week 6]

## 📊 Benchmark Results
[Tables — added Week 7-8]

## 🗂️ Dataset
- UA-DETRAC (140,131 pre-annotated highway images)
- NGSIM (US Federal Highway trajectory data)
- Custom dashcam clips for demo

## ⚖️ License
MIT
