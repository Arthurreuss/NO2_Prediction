import yaml

from src.features.preprocess import PreProcessingPipeline
from src.utils.cfg import load_config

if __name__ == "__main__":
    config = load_config("configs/config_training.yaml")
    pipeline = PreProcessingPipeline(cfg=config, norm_method="standard")
    pipeline.preprocess()
