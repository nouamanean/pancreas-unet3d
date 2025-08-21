import yaml
from src.data.preprocessing import IRMPreprocessor

with open("config/config.yaml", "r") as f:
    full_config = yaml.safe_load(f)
    config = full_config["preprocessing"]

preproc = IRMPreprocessor(config)
preproc.process_all()
preproc.extract_patches()
