import joblib
import mlflow
import pandas as pd
import torch

from src.utils.device import get_device


class ModelWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.scaler = joblib.load(context.artifacts["scaler"])
        device = get_device()
        self.model = torch.load(context.artifacts["pytorch_model"], map_location=device)
        self.model.eval()

    def predict(self, context, model_input):
        """
        Expects 'model_input' to be an ALREADY SCALED DataFrame or Numpy array.
        """
        if isinstance(model_input, pd.DataFrame):
            data = model_input.values
        else:
            data = model_input

        tensor_input = torch.tensor(data, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            output = self.model(tensor_input)

        return output.numpy()
