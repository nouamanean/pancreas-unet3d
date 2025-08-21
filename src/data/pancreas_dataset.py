import os
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd


class PancreasPatchDataset(Dataset):
    def __init__(self, patch_dir_img, patch_dir_label, metadata_path, transform=None):
        """
        Dataset pour charger les patchs pancréatiques en .npz
        """
        self.patch_dir_img = patch_dir_img
        self.patch_dir_label = patch_dir_label
        self.metadata = pd.read_csv(metadata_path)
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        img_path = os.path.join(self.patch_dir_img, row["patient_id"], row["patch_file"])
        lab_path = os.path.join(self.patch_dir_label, row["patient_id"], row["patch_file"])

        with np.load(img_path) as npz:
            img = npz["patch"]

        with np.load(lab_path) as npz:
            label = npz["patch"]

        # Convert to tensors
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0) / 255.0  # [1, D, H, W]
        label = torch.tensor(label, dtype=torch.float32).unsqueeze(0)  # [1, D, H, W]

        if self.transform:
            img, label = self.transform(img, label)

        return img, label
