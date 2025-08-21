# scripts/train.py

import sys
import os
import yaml

# Add the parent folder to the PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.training.train_ import train_model

# Load the config
with open("config/config.yaml", "r") as f:
    cfg = yaml.safe_load(f)["dataset"]

# Start training
train_model(cfg)

