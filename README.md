---
title: NO2 Forecasting
emoji: 🌍
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.52.2
python_version: 3.12
app_file: streamlit_app/app.py
pinned: false
---

# NO₂ Air Quality Forecasting Pipeline for Utrecht

> **Real-time forecasting of Nitrogen Dioxide (NO₂) and co-pollutants using Deep Learning & MLOps.**

## Overview
This project implements an end-to-end machine learning pipeline to predict air quality levels (specifically NO₂, PM10, PM2.5, and Ozone) in Utrecht. It leverages **Recurrent Neural Networks (GRUs)** and a fully automated MLOps workflow—from hyperparameter optimization to deployment on Hugging Face Spaces.

## Key Features
* **Data Source:** Fetches real-time weather and air quality data via the **Open-Meteo API**.
* **Advanced Modeling:** Implements **GRU, Hierarchical GRU (HGRU), and Multi-Output GRU** architectures using PyTorch.
* **Smart Training:** Hyperparameter optimization using **Optuna** and experiment tracking with **MLflow**.
* **Interpretability:** Includes feature importance and temporal saliency analysis to understand model decisions.
* **Automated Ops:** All Models are trained locally and migrated to **DagsHub** (Remote MLflow Registry).
    * **GitHub Actions** trigger scheduled inference pipelines.
    * Results are visualized in a **Streamlit** dashboard hosted on **Hugging Face Spaces**.

## 🛠️ Tech Stack
* **Core:** Python 3.12, PyTorch, Pandas, NumPy
* **MLOps:** MLflow, DagsHub, Optuna, GitHub Actions
* **Deployment:** Streamlit, Docker, Hugging Face Spaces
* **Analysis:** Jupyter Notebooks (Data Drift & Feature Selection)

## Project Structure

```bash
.
├── configs/                  
│   ├── config_analysis.yaml
│   ├── config_deployment.yaml
│   └── config_training.yaml
├── data/                
│   ├── analysis/
│   ├── deployment/
│   └── training/
├── deployment/              
│   ├── migration/
│   │   ├── migrate.py
│   │   └── model_scaler_wrapper.py
│   ├── compute_metrics.py
│   └── run_inference.py
├── notebooks/            
│   ├── data_drift.ipynb
│   ├── feature_selection.ipynb
│   └── plot.ipynb
├── results/                
├── scripts/                   
│   ├── hp_optimization/      
│   ├── run_model/            
│   ├── interpretability.py
│   └── run_preprocessing.py
├── src/                     
│   ├── data/                
│   ├── features/            
│   ├── interpret/            
│   ├── models/                
│   ├── training/              
│   └── utils/               
├── streamlit/                 
│   ├── ui/
│   │   ├── admin_view.py
│   │   └── user_view.py
│   ├── utils/
│   └── app.py
├── Dockerfile
├── makefile
├── pyproject.toml
├── README.md
├── requirements.txt
└── uv.lock
```
## How to Run Locally

### 1. Prerequisites

Ensure you have Python installed. Clone the repository and install dependencies:

```bash
git clone https://github.com/Arthurreuss/NO2_Prediction.git
cd NO2_Prediction
pip install -r requirements.txt

```

### 2. Environment Setup

Create a `.env` file in the root directory to store your credentials (only needed for deployment):

```env
MLFLOW_TRACKING_URI=https://dagshub.com/username/repo.mlflow
MLFLOW_TRACKING_USERNAME=your_username
MLFLOW_TRACKING_PASSWORD=your_token
DISCORD_WEBHOOK_URL=your_webhook_url
HF_ACCESS_TOKEN=token
HF_USERNAME=username
HF_SPACE_NAME=NO2_Prediction

```

### 3. Training Pipeline

Before training, run the preprocessing pipeline to clean and prepare the data:

```bash
python -m scripts.run_preprocessing

```

Next, you can run hyperparameter optimization (using Optuna) or train a standard model:

```bash
# Run Optimization
python -m scripts.hp_optimization.optimize_gru

# Train Model
python -m scripts.run_model.run_gru

```

### 4. Local Dashboard

Launch the Streamlit app locally to view predictions and model performance:

```bash
streamlit run streamlit_app/app.py

```

### 5. Deployment

Deployment consists of two stages: promoting models to the remote registry and automating inference via GitHub Actions.

#### A. Model Migration

Run the migration script to promote your locally trained models to the remote DagsHub registry.

> **Note:** The script will only migrate models that are **registered** in your local MLflow and tagged with the **`@production`** alias.

```bash
uv run python -m deployment.migration.migrate

```

#### B. Automated Architecture (CI/CD)

This project uses a lightweight, automated deployment strategy to minimize frontend resource usage.

* **Automated Inference:** A **GitHub Actions** workflow runs hourly, executing inference only on models tagged with the **`@production`** alias in the DagsHub registry.
* **Pre-computed Metrics:** All performance metrics are calculated within the GitHub runner, ensuring the hosted application remains fast and lightweight.
* **Live Visualization:** Updated predictions and metrics are pushed directly to **Hugging Face**, where the Streamlit app simply visualizes the data.

**How to Deploy Yourself:**

1. Create a **Hugging Face Space**.
2. Update the GitHub repository secrets with your Hugging Face credentials.
3. The workflow in `.github/workflows` will automatically handle hourly inference and data syncing.Deployment Architecture

