import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from sklearn.model_selection import train_test_split
from src.data.pancreas_dataset import PancreasPatchDataset
from src.data.preprocessing import IRMPreprocessor
from src.models.unet3D import UNet3D
from src.training.train_ import train_model

print("🔄 Starting data split...")

# Load metadata
df = pd.read_csv("data/processed/patches/metadata.csv")

# Split into train (70%) and temp (30%)
train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)

# ✅ Limit to max 5000 samples
train_df = train_df[:5000]

# Split temp into val (15%) and test (15%)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# Save CSV files
train_df.to_csv("data/processed/patches/train.csv", index=False)
val_df.to_csv("data/processed/patches/val.csv", index=False)
test_df.to_csv("data/processed/patches/test.csv", index=False)

print("✅ Split completed:")
print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
