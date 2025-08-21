import streamlit as st
import torch
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from dataset_pancreas.pancreas_dataset import PancreasPatchDataset
from models.unet3D import UNet3D
st.title("Visualisation des patchs pancréatiques")

# 📁 Dossier des patchs compressés
PATCH_DIR_IMG = "Resample_data/patches/irm_t2"
PATCH_DIR_LABEL = "Resample_data/patches/labels"
METADATA_PATH = "Resample_data/patches/test.csv"
MODEL_PATH = "models/unet3d_trained.pth"



# Charger modèle
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet3D()
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# Metadata
df = pd.read_csv(METADATA_PATH)
st.title("🧠 Visualisation des prédictions U-Net 3D")

selected = st.selectbox("📦 Choisir un patch", df.index)

row = df.iloc[selected]
patch_file = row["patch_file"]
patient_id = row["patient_id"]

img_path = os.path.join(PATCH_DIR_IMG, patient_id, patch_file)
label_path = os.path.join(PATCH_DIR_LABEL, patient_id, patch_file)

img = np.load(img_path)["patch"]
label = np.load(label_path)["patch"]

# Prédiction
with torch.no_grad():
    input_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
    pred = model(input_tensor.to(DEVICE))
    pred_np = pred.squeeze().cpu().numpy() > 0.5

# Slice
depth = img.shape[0]
slice_idx = st.slider("🩻 Slice Z", 0, depth - 1, depth // 2)

# Affichage
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(img[slice_idx], cmap="gray")
axes[0].set_title("IRM")
axes[1].imshow(label[slice_idx], cmap="gray")
axes[1].set_title("Masque Réel")
axes[2].imshow(pred_np[slice_idx], cmap="hot", alpha=0.8)
axes[2].set_title("Prédiction")
for ax in axes: ax.axis("off")
st.pyplot(fig)
