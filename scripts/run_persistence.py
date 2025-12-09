import mlflow
import yaml

from src.data.dataloaders import make_dataloader
from src.models.persistence import PersistenceModel
from src.utils.eval import evaluate_persistence


def main():
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    mlflow.set_experiment("NO2_Forecasting")

    data_cfg = cfg["data"]
    train_path = data_cfg["train_path"]
    val_path = data_cfg["val_path"]
    test_path = data_cfg["test_path"]
    feature_cols = data_cfg["feature_cols"]
    target_col = data_cfg["target_col"]
    input_length = data_cfg["input_length"]
    horizon = data_cfg["horizon"]

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

        print("Evaluating persistence on val set...")
        val_metrics = evaluate_persistence(val_loader, model, horizon)
        print(val_metrics)

        print("Evaluating persistence on test set...")
        test_metrics = evaluate_persistence(test_loader, model, horizon)
        print(test_metrics)

        mlflow.log_metric("val_rmse", float(val_metrics.get("rmse", float("nan"))))
        mlflow.log_metric("val_smape", float(val_metrics.get("smape", float("nan"))))
        mlflow.log_metric("test_rmse", float(test_metrics.get("rmse", float("nan"))))
        mlflow.log_metric("test_smape", float(test_metrics.get("smape", float("nan"))))

        mlflow.log_artifact("config.yaml")


if __name__ == "__main__":
    main()
