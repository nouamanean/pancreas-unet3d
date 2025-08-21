import os
import torch
import yaml
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path

# Internal imports
from src.data.pancreas_dataset import PancreasPatchDataset
from src.models.unet3D import UNet3D

def load_config(config_path="config/config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def dice_score(pred, target):
    pred_bin = (pred > 0.5).float()
    intersection = (pred_bin * target).sum()
    dice = (2. * intersection) / (pred_bin.sum() + target.sum() + 1e-8)
    return dice.item()

def evaluate_model(cfg):
    print("🚀 Loading model for evaluation...")
    model = UNet3D(n_channels=1, n_classes=1)
    model.load_state_dict(torch.load(cfg["model_path"], map_location="cpu"))
    model.eval()

    print("📦 Loading test dataset...")
    test_dataset = PancreasPatchDataset(
        patch_dir_img=cfg["img_dir"],
        patch_dir_label=cfg["label_dir"],
        metadata_path=cfg["test_csv"]
    )
    test_loader = DataLoader(test_dataset, batch_size=1)

    dice_scores = []
    with torch.no_grad():
        for img, mask in test_loader:
            pred = model(img)
            dice = dice_score(pred, mask)
            dice_scores.append(dice)

    print(f"\n✅ Average Dice on test set: {np.mean(dice_scores):.4f}")

if __name__ == "__main__":
    # Load configuration
    config = load_config()
    eval_cfg = config["evaluation"]

    # Check required paths
    for key in ["test_csv", "img_dir", "label_dir", "model_path"]:
        if not os.path.exists(eval_cfg[key]):
            raise FileNotFoundError(f"❌ Missing file: {eval_cfg[key]}")

    # Run evaluation
    evaluate_model(eval_cfg)
