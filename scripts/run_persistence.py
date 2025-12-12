import joblib
import mlflow
import pandas as pd
import yaml

from src.data.dataloaders import make_dataloader
from src.models.persistence import PersistenceModel
from src.utils.eval import evaluate_persistence


def main():
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    mlflow_cfg = cfg["training"]["mlflow"]
    mlflow.set_experiment(mlflow_cfg["experiment_name"])

    data_cfg = cfg["data"]
    train_path = data_cfg["train_path"]
    val_path = data_cfg["val_path"]
    test_path = data_cfg["test_path"]
    feature_cols = data_cfg["feature_cols"]
    target_col = data_cfg["target_col"]
    input_length = data_cfg["input_length"]
    horizon = data_cfg["horizon"]

    scaler_path = data_cfg.get("scaler_path", "scaler.pkl")
    scaler = joblib.load(scaler_path)

    if "numeric_cols" in data_cfg and data_cfg["numeric_cols"]:
        numeric_cols = data_cfg["numeric_cols"]
    else:
        df_train = pd.read_parquet(train_path)
        numeric_cols = df_train.select_dtypes(include=["number"]).columns.tolist()

    train_loader = make_dataloader(
        train_path,
        feature_cols,
        target_col,
        input_length,
        horizon,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
    )
    val_loader = make_dataloader(
        val_path,
        feature_cols,
        target_col,
        input_length,
        horizon,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
    )
    test_loader = make_dataloader(
        test_path,
        feature_cols,
        target_col,
        input_length,
        horizon,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
    )

    target_idx = feature_cols.index(target_col)
    model = PersistenceModel(target_feature_index=target_idx)

    with mlflow.start_run(run_name="Persistence_baseline"):
        mlflow.log_param("model_type", "persistence")
        mlflow.log_param("horizon", horizon)
        mlflow.log_param("input_length", input_length)
        mlflow.log_param("target_col", target_col)
        mlflow.log_param("features", ",".join(feature_cols))
        mlflow.log_param("scaler_path", scaler_path)
        mlflow.log_param("numeric_cols_count", len(numeric_cols))

        print("Evaluating persistence on val set (SCALED)...")
        val_metrics_scaled = evaluate_persistence(val_loader, model, horizon)
        print(val_metrics_scaled)

        print("Evaluating persistence on val set (DENORMALIZED)...")
        val_metrics = evaluate_persistence(
            val_loader,
            model,
            horizon,
            scaler=scaler,
            numeric_cols=numeric_cols,
            target_col=target_col,
        )
        print(val_metrics)

        print("Evaluating persistence on test set (SCALED)...")
        test_metrics_scaled = evaluate_persistence(test_loader, model, horizon)
        print(test_metrics_scaled)

        print("Evaluating persistence on test set (DENORMALIZED)...")
        test_metrics = evaluate_persistence(
            test_loader,
            model,
            horizon,
            scaler=scaler,
            numeric_cols=numeric_cols,
            target_col=target_col,
        )
        print(test_metrics)

        mlflow.log_metric(
            "val_rmse_scaled", float(val_metrics_scaled.get("rmse", float("nan")))
        )
        mlflow.log_metric(
            "val_smape_scaled", float(val_metrics_scaled.get("smape", float("nan")))
        )
        mlflow.log_metric(
            "test_rmse_scaled", float(test_metrics_scaled.get("rmse", float("nan")))
        )
        mlflow.log_metric(
            "test_smape_scaled", float(test_metrics_scaled.get("smape", float("nan")))
        )

        mlflow.log_metric("val_rmse", float(val_metrics.get("rmse", float("nan"))))
        mlflow.log_metric("val_smape", float(val_metrics.get("smape", float("nan"))))
        mlflow.log_metric("test_rmse", float(test_metrics.get("rmse", float("nan"))))
        mlflow.log_metric("test_smape", float(test_metrics.get("smape", float("nan"))))

        mlflow.log_artifact("config.yaml")
        mlflow.log_artifact(scaler_path)


if __name__ == "__main__":
    main()
