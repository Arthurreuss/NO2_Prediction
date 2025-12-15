import yaml

from scripts.preprocess import PreProcessingPipeline

if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    pipeline = PreProcessingPipeline(cfg=config, norm_method="standard")
    pipeline.preprocess()
