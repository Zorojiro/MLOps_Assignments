# MLOps Assignments


This repository contains assignments focused on MLOps concepts and tools, including KServe, Kubeflow, Streamlit, and Observability. Each assignment folder contains source code, deployment files, and scripts tailored to specific use-cases in the MLOps lifecycle.

---

## Directory Overview

### 🔹`Kserve/`


A machine learning model deployed using **KServe** on Kubernetes for predicting crop diseases.
- Includes: `disease_prediction.py`, `inference.py`, `deployment.yaml`, `Dockerfile`, trained model & encoders.
- Features: Dockerization, custom inference logic, KServe REST deployment.

---

### 🔹`Kuberflow/`



A Kubeflow pipeline to predict **crop disease risk** based on environmental features.
- Shows end-to-end workflow: data processing, model inference, and visualization using pipeline components.

---

### 🔹 `Observability/`


Basic observability assignment demonstrating **Elasticsearch + Kibana** setup.
- Includes configuration and data indexing steps to monitor logs/metrics.

---

### 🔹 `Python/`

#### • `Assignment_1/`
Python script showcasing basic automation or scripting tasks.
- Useful for learning scripting basics before full ML pipelines.

#### • `Assignment_2/`
Flask-based web app featuring:
- `images/`, `templates/`, `static/` — assets for web frontend
- Web interface for prediction or visualization (likely related to ML model)

---

### 🔹 `Streamlit/`

#### • `Assignment1_Visualizing CSV Data/`
Streamlit-based dashboard to visualize uploaded CSV files using interactive charts.
- Allows users to upload any dataset and explore it visually.

#### • `Assignment2_Weather_App/`
A weather dashboard built with Streamlit fetching real-time data from an external API.
- Users can input a city and get weather details like temperature, humidity, etc.

#### • `Assignment3_Sentiment_Analysis/`
Sentiment analysis web app using NLP libraries and Streamlit UI.
- Input a sentence and get sentiment prediction (Positive/Negative).

## 🛠 Requirements

- Docker, Kubernetes
- KServe, Kubeflow Pipelines
- Python 3.8+
- Streamlit, Flask
- Elasticsearch (for Observability)

---

