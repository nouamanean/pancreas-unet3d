# src/data/preprocessing.py
import os
import numpy as np
import SimpleITK as sitk
import pandas as pd
from tqdm import tqdm
from sklearn.cluster import KMeans
from pathlib import Path

class IRMPreprocessor:
    def __init__(self, config):
        self.config = config
        self.input_dir = config["input_dir"]
        self.label_dir = config["label_dir"]
        self.resample_dir = config["resample_dir"]
        self.resample_t2 = os.path.join(self.resample_dir, config["resample_t2"])
        self.resample_label = os.path.join(self.resample_dir, config["resample_label"])
        self.patch_dir = os.path.join(self.resample_dir, config["output_patch"])
        self.patch_t2 = os.path.join(self.patch_dir, config["output_patch_t2"])
        self.patch_lab = os.path.join(self.patch_dir, config["output_patch_lab"])
        self.target_shape = tuple(config["target_shape"])
        self.new_spacing = tuple(config["new_spacing"])

        self._create_directories()

    def _create_directories(self):
        for path in [
            self.input_dir, self.label_dir, self.resample_dir,
            self.resample_t2, self.resample_label,
            self.patch_dir, self.patch_t2, self.patch_lab
        ]:
            os.makedirs(path, exist_ok=True)

    def list_files(self, directory):
        return [os.path.join(directory, f) for f in os.listdir(directory)]

    def resample_image(self, image_path, is_label=False, normalize=False):
        image = sitk.ReadImage(image_path)
        original_spacing = image.GetSpacing()
        original_size = image.GetSize()
        new_size = [int(round(osz * ospc / tspc)) for osz, ospc, tspc in zip(original_size, original_spacing, self.new_spacing)]

        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(self.new_spacing)
        resampler.SetSize(new_size)
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear)

        resampled = resampler.Execute(image)

        if normalize and not is_label:
            arr = sitk.GetArrayFromImage(resampled).astype(np.float32)
            arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255
            arr = arr.astype(np.uint8)
            resampled = sitk.GetImageFromArray(arr)
        return resampled

    def extract_patches(self, patch_size=(64, 128, 128), stride=(32, 64, 64), background_ratio=0.1):
        image_paths = self.list_files(self.resample_t2)
        label_paths = self.list_files(self.resample_label)
        metadata = []

        for img_path, lbl_path in tqdm(zip(image_paths, label_paths), total=len(image_paths), desc="Patch extraction"):
            pid = Path(img_path).stem.split("_resampled")[0]
            img = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
            lbl = sitk.GetArrayFromImage(sitk.ReadImage(lbl_path))

            z_max, y_max, x_max = img.shape
            pz, py, px = patch_size
            sz, sy, sx = stride
            count = 0

            for z in range(0, z_max - pz + 1, sz):
                for y in range(0, y_max - py + 1, sy):
                    for x in range(0, x_max - px + 1, sx):
                        patch_img = img[z:z+pz, y:y+py, x:x+px]
                        patch_lbl = lbl[z:z+pz, y:y+py, x:x+px]
                        has_tumor = np.sum(patch_lbl) > 0
                        if has_tumor or np.random.rand() < background_ratio:
                            fname = f"{pid}_patch_{count:04d}.npz"
                            np.savez_compressed(os.path.join(self.patch_t2, pid, fname), patch=patch_img)
                            np.savez_compressed(os.path.join(self.patch_lab, pid, fname), patch=patch_lbl)
                            metadata.append({
                                "patient_id": pid,
                                "patch_file": fname,
                                "has_tumor": int(has_tumor),
                                "z": z, "y": y, "x": x
                            })
                            count += 1

        pd.DataFrame(metadata).to_csv(os.path.join(self.patch_dir, "metadata.csv"), index=False)

    def process_all(self):
        images = self.list_files(self.input_dir)
        labels = self.list_files(self.label_dir)

        for img_path, lbl_path in tqdm(zip(images, labels), total=len(images), desc="Resampling"):
            pid = Path(img_path).stem.replace(".nii.gz", "")
            out_img = os.path.join(self.resample_t2, f"{pid}_resampled.nii.gz")
            out_lbl = os.path.join(self.resample_label, f"{pid}_resampled.nii.gz")

            if not os.path.exists(out_img):
                img_res = self.resample_image(img_path, is_label=False, normalize=True)
                sitk.WriteImage(img_res, out_img)
            if not os.path.exists(out_lbl):
                lbl_res = self.resample_image(lbl_path, is_label=True)
                sitk.WriteImage(lbl_res, out_lbl)
