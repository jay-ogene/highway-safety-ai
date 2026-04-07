# 🚗 AI Highway Safety & Incident Detection Platform



> Real-time vehicle detection, tracking, and incident classification 
> deployed on Google Cloud Platform.

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8s-0.833_mAP@0.5-green)](https://ultralytics.com)
[![GCP](https://img.shields.io/badge/GCP-Cloud_Run_Deployed-orange)](https://cloud.google.com)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

**Live API:** https://highway-safety-api-720304622514.us-central1.run.app/docs  
**Dashboard:** *(Streamlit Cloud URL once deployed)*  
**Author:** Jude Ogene — M.S. Computer Science, Kennesaw State University (Dec 2024)

---

## What It Does

This platform processes overhead highway camera footage in real time,
detecting vehicles, tracking their movement across frames, and 
classifying dangerous incidents before they become crashes.

**Validated against 8.3 million real vehicle observations** 
from the NGSIM I-80 dataset (US DOT, Berkeley CA).

---

## Results

| Metric | Value |
|--------|-------|
| Detection model | YOLOv8s |
| mAP@0.5 | **0.833** |
| mAP@0.5-95 | 0.650 |
| Training data | 140,000 frames (UA-DETRAC) |
| Training hardware | Dual T4 GPU (Kaggle) |
| Incidents logged | 1,553 (100-frame test sequence) |
| Critical alerts | 201 (TTC < 1.5s) |
| Min TTC recorded | 0.13 seconds |
| NGSIM validation | ✅ 8.3M observations |
| UA-DETRAC baseline (YOLOv3) | 76% AP |
| **Improvement over baseline** | **+7.3 mAP points** |

---

## Incident Types Detected

| Type | Method | Threshold |
|------|--------|-----------|
| Near-miss | Time-To-Collision | TTC < 3.0s |
| Critical near-miss | TTC | TTC < 1.5s |
| Tailgating | Gap + speed + persistence | < 10m at > 20 km/h for 5+ frames |
| Sudden braking | Deceleration rate | > 15 km/h/s over 3 frames |
| Wrong-way driving | Flow direction | Opposing dominant flow for 8+ frames |
| Stopped vehicle | Speed threshold | < 3 km/h for 10+ frames |

---

## Architecture


---

## Tech Stack

**Computer Vision & ML**
- YOLOv8s (Ultralytics) — vehicle detection
- ByteTrack + Kalman filter — multi-object tracking  
- OpenCV — frame processing and annotation
- PyTorch 2.2 — model inference

**Cloud Infrastructure (GCP)**
- Cloud Run — containerised API deployment
- Pub/Sub — real-time incident streaming
- BigQuery — incident data warehouse
- Cloud Storage — model weight storage
- Artifact Registry — Docker container registry
- Vertex AI — production model training

**Backend & API**
- FastAPI — REST API with auto-documentation
- Uvicorn — ASGI server
- Pydantic — request/response validation
- Docker — containerisation

**Data & Validation**
- UA-DETRAC — 140,000 labeled highway frames
- NGSIM I-80 — 8.3M real vehicle observations (US DOT)
- Pandas, NumPy, SciPy — data processing
- Weights & Biases — experiment tracking

**Dashboard**
- Streamlit — live incident dashboard
- Plotly — interactive charts

---

## Project Structure

---

## Quick Start

**Prerequisites:** Python 3.11, conda, GCP account with credentials
```bash
# Clone and set up environment
git clone https://github.com/jay-ogene/highway-safety-ai.git
cd highway-safety-ai
conda create -n highway-safety python=3.11
conda activate highway-safety
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your GCP credentials

# Run the pipeline on a UA-DETRAC sequence
python run_pipeline.py

# Start the dashboard
streamlit run dashboard/app.py

# Start the API locally
uvicorn ingestion.api:app --reload --port 8000
```

---

## API Endpoints

Base URL: `https://highway-safety-api-720304622514.us-central1.run.app`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | System status + GCP connectivity |
| `/incidents` | GET | Query incidents with filters |
| `/incidents/summary` | GET | Aggregated statistics |
| `/analyze` | POST | Trigger pipeline on a sequence |
| `/docs` | GET | Interactive Swagger UI |

---

## Validation

Speed estimates and alert thresholds validated against 
**8,305,244 real vehicle observations** from the 
[NGSIM I-80 dataset](https://ops.fhwa.dot.gov/trafficanalysistools/ngsim.htm) 
(US Department of Transportation):

- Speed range: pipeline covers full NGSIM congested highway distribution ✅
- TTC thresholds: 11.5% of real I-80 vehicles operate at CRITICAL levels ✅  
- Tailgating threshold (10m): correctly flags 9.7% of real following pairs ✅
- Lane analysis: inner lanes show 6.7× higher tailgating rates ✅

---

## Background

Built as a 12-week portfolio project demonstrating end-to-end 
ML engineering for H-1B visa sponsorship applications.

**Author:** Jude Ogene  
**Education:** M.S. Computer Science (AI & Data Science), 
Kennesaw State University, December 2024  
**Location:** Atlanta, GA  
**GitHub:** [@jay-ogene](https://github.com/jay-ogene)  
**Status:** Seeking H-1B sponsorship for AI Engineer roles

---

## License

MIT License — see [LICENSE](LICENSE) for details.

Dataset credits:
- UA-DETRAC: Wen et al., CVIU 2020
- NGSIM: US Department of Transportation