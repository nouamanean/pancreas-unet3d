# Pancreas Segmentation with 3D U-Net

This project implements a full deep learning pipeline for pancreas tumor segmentation using **3D U-Net**.  
The workflow includes preprocessing, patch extraction, dataset splitting, model training, and evaluation.

---

## 📂 Project structure

config/ # configuration files (YAML)
data/processed/patches/ # preprocessed patches and metadata
results/checkpoints/ # model checkpoints per epoch
results/best_model.pth # best model (lowest validation loss)
scripts/preprocess.py # run preprocessing and patch extraction
scripts/split.py # split data into train/val/test
src/data/preprocessing.py # MRI preprocessing and patch extraction
src/data/pancreas_dataset.py # Dataset loader for patches
src/models/unet3D.py # 3D U-Net implementation
src/training/train_.py # Training loop
src/training/evaluate.py # Model evaluation
main.py # Pipeline runner


---


🚀 Usage
1. Preprocessing and Patch Extraction

python scripts/preprocess.py
This will resample MRI scans and extract 3D patches.

2. Split Train/Val/Test

Generates:

train.csv
val.csv
test.csv

3. Training
python main.py

This runs the pipeline and trains the 3D U-Net.

Models are saved in:
results/checkpoints/
results/best_model.pth

4. Evaluation
python src/training/evaluate.py

Computes the average Dice score on the test set.

📊 Outputs

Training and validation losses per epoch

Checkpoints saved every epoch

Best model automatically saved when validation improves

Evaluation Dice score on test dataset

🛠️ Requirements

Python 3.9+
PyTorch
SimpleITK
scikit-learn
pandas
tqdm
pyyaml

✨ Features

Full 3D U-Net implementation

Medical image preprocessing with SimpleITK

Patch extraction with tumor/background balancing

Train/validation/test split

Training with checkpointing and best model saving

Evaluation with Dice score

📌Notes

Input MRI scans must be in .nii.gz format

Run preprocessing before training

Adjust batch_size, learning_rate, and num_epochs in config.yaml depending on hardware

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
