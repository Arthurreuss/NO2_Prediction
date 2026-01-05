---
title: NO2 Forecasting
emoji: 🌍
colorFrom: blue
colorTo: green
sdk: streamlit
app_file: streamlit/app.py
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
│   └── deployment_pipeline.py
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

Create a `.env` file in the root directory to store your credentials (required for MLflow/DagsHub tracking):

```env
MLFLOW_TRACKING_URI=https://dagshub.com/username/repo.mlflow
MLFLOW_TRACKING_USERNAME=your_username
MLFLOW_TRACKING_PASSWORD=your_token
```

### 3. Training & Optimization

Before training, run the preprocessing pipeline to clean and prepare the data:

```bash
python -m scripts.run_preprocessing.py
```

Next, you can run hyperparameter optimization (using Optuna) or train a standard model.

**To run the GRU optimization or GRU:**

```bash
python -m scripts.hp_optimization.optimize_gru
python -m scripts.run_model.run_gru
```


### 4. Run the Dashboard

Launch the Streamlit app to view predictions and model performance:

```bash
streamlit run streamlit/app.py
```

### 5. Management & Deployment Tools

* **Model Migration:**
Run the migration script to promote your locally trained models to the remote DagsHub registry.
> **Note:** The script will only migrate models that are **registered** in your local MLflow and tagged with the **`@production`** alias.
```bash
uv run python -m deployment.migration.migrate
````

* **MLflow UI:** To inspect registered models and experiment runs locally, use:
```bash
mlflow ui
```



> **Note on Deployment:**
> A **GitHub Actions** workflow triggers every hour. It runs the inference pipeline using only models marked with the **`@production`** alias in the DagsHub registry and pushes the updated results to **Hugging Face**, where the hosted Streamlit application displays the live results.
>
> You can find the specific workflow configuration under **`.github/workflows`** if you are interested in how the automated inference and push to Hugging Face is implemented.