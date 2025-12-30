import os
from pathlib import Path

import mlflow
import torch
from dotenv import load_dotenv

from deployment.migration.model_scaler_wrapper import ModelWrapper
from src.utils.cfg import load_config
from src.utils.device import get_device
from src.utils.helper import load_torch_model_from_registry


def migrate():
    load_dotenv()
    device = get_device()
    cfg = load_config("config_deployment.yaml")
    migration_cfg = cfg["deployment"]["migration"]
    dags_uri = migration_cfg["dags_uri"]
    local_scaler_path = migration_cfg["scaler_path"]

    print("--- STEP 1: LOADING FROM LOCAL REGISTRY ---")

    models_map = {
        "NO2_Forecasting_GRU": load_torch_model_from_registry("gru_best_model", device),
        "NO2_Forecasting_MultiGRU": load_torch_model_from_registry(
            "multigru_best_model", device
        ),
        "NO2_Forecasting_HGRU": load_torch_model_from_registry(
            "hgru_best_model", device
        ),
    }

    print("\n--- STEP 2: SWITCHING TO DAGSHUB ---")
    print(f"Tracking URI set to: {mlflow.get_tracking_uri()}")
    mlflow.set_tracking_uri(dags_uri)
    mlflow.set_registry_uri(dags_uri)
    mlflow.set_experiment("NO2_Forecasting_Migration")

    print("\n--- STEP 3: UPLOADING TO REMOTE ---")
    for reg_name, model_obj in models_map.items():
        print(f"Preparing {reg_name}...")
        torch.save(model_obj, "temp_weights.pt")
        artifacts = {"scaler": local_scaler_path, "pytorch_model": "temp_weights.pt"}

        with mlflow.start_run(run_name=f"Migrate_{reg_name}"):
            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=ModelWrapper(),
                artifacts=artifacts,
                registered_model_name=reg_name,
            )
            print(f"Registered {reg_name}!")

    if os.path.exists("temp_weights.pt"):
        os.remove("temp_weights.pt")
        print("Cleaned up temporary files.")


if __name__ == "__main__":
    migrate()
