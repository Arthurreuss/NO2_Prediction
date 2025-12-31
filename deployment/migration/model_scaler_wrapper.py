from typing import Any, Union

import joblib
import mlflow
import pandas as pd
import torch

from src.utils.device import get_device


class ModelWrapper(mlflow.pyfunc.PythonModel):
    """MLflow PyFunc wrapper for a torch forecasting model with an associated scaler.

    The wrapper loads:
      - a fitted scaler from an MLflow artifact path
      - a serialized PyTorch model from an MLflow artifact path

    The `predict` method expects inputs that are already scaled.
    """

    def load_context(self, context: Any) -> None:
        """Load artifacts from the MLflow model context.

        This method is called by MLflow when the PyFunc model is loaded.

        Args:
            context: MLflow context object providing access to model artifacts.
                Expected to contain:
                  - context.artifacts["scaler"]
                  - context.artifacts["pytorch_model"]
        """
        self.scaler = joblib.load(context.artifacts["scaler"])
        device = get_device()
        self.model = torch.load(context.artifacts["pytorch_model"], map_location=device)
        self.model.eval()

    def predict(self, context: Any, model_input: Union[pd.DataFrame, Any]) -> Any:
        """Run inference on already-scaled input data.

        The input is expected to be either a pandas DataFrame or a numpy-like array
        that is already scaled/normalized. If a DataFrame is provided, its `.values`
        are used as the numeric input matrix.

        Args:
            context: MLflow prediction context (unused).
            model_input: Already-scaled input features as either:
                - a pandas DataFrame (values are extracted), or
                - a numpy-like array of shape [N, F] (or compatible)

        Returns:
            The model output as a numpy array.
        """
        if isinstance(model_input, pd.DataFrame):
            data = model_input.values
        else:
            data = model_input

        tensor_input = torch.tensor(data, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            output = self.model(tensor_input)

        return output.numpy()
