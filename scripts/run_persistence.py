import joblib
import mlflow
import numpy as np
import yaml

from src.data.dataloaders import make_dataloader
from src.models.persistence import PersistenceModel
from src.utils.eval import evaluate_persistence


def make_target_inverse_transform(scaler, feature_cols, target_idx):
    """
    Returns a function inv(y_scaled) that maps (N, horizon) scaled target -> original units.
    Handles two cases:
      A) scaler fitted on ALL features (n_features_in_ == len(feature_cols))
      B) scaler fitted on target only (n_features_in_ == 1)
    """
    n_features = getattr(scaler, "n_features_in_", None)

    def inv(y_scaled):
        y_scaled = np.asarray(y_scaled)
        orig_shape = y_scaled.shape
        y2d = y_scaled.reshape(-1, 1)  # (N*horizon, 1)

        if n_features == len(feature_cols):
            X = np.zeros((y2d.shape[0], len(feature_cols)), dtype=float)
            X[:, target_idx] = y2d[:, 0]
            X_inv = scaler.inverse_transform(X)
            y_inv = X_inv[:, target_idx].reshape(orig_shape)
            return y_inv

        if n_features == 1 or n_features is None:
            y_inv = scaler.inverse_transform(y2d).reshape(orig_shape)
            return y_inv

        raise ValueError(
            f"Unsupported scaler.n_features_in_={n_features}. "
            f"Expected 1 or {len(feature_cols)}."
        )

    return inv


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

    # ---- load scaler ----
    scaler_path = cfg["data"].get("scaler_path", "scaler.pkl")
    scaler = joblib.load(scaler_path)

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

    inv_target = make_target_inverse_transform(scaler, feature_cols, target_idx)

    with mlflow.start_run(run_name="Persistence_baseline"):
        mlflow.log_param("model_type", "persistence")
        mlflow.log_param("horizon", horizon)
        mlflow.log_param("input_length", input_length)
        mlflow.log_param("target_col", target_col)
        mlflow.log_param("features", ",".join(feature_cols))
        mlflow.log_param("scaler_path", scaler_path)

        print("Evaluating persistence on val set (SCALED)...")
        val_metrics_scaled = evaluate_persistence(val_loader, model, horizon)
        print(val_metrics_scaled)

        print("Evaluating persistence on val set (DENORMALIZED)...")
        val_metrics = evaluate_persistence(
            val_loader, model, horizon, inverse_transform=inv_target
        )
        print(val_metrics)

        print("Evaluating persistence on test set (SCALED)...")
        test_metrics_scaled = evaluate_persistence(test_loader, model, horizon)
        print(test_metrics_scaled)

        print("Evaluating persistence on test set (DENORMALIZED)...")
        test_metrics = evaluate_persistence(
            test_loader, model, horizon, inverse_transform=inv_target
        )
        print(test_metrics)

        # log both
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
        # optional: log the scaler used
        mlflow.log_artifact(scaler_path)


if __name__ == "__main__":
    main()
