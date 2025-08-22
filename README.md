<h1 align="center">PANCREAS-UNET3D</h1>

<p align="center">
  <em>Transforming Medical Imaging with Precision and Power</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/last%20commit-today-brightgreen" />
  <img src="https://img.shields.io/badge/python-100%25-blue" />
  <img src="https://img.shields.io/badge/languages-1-lightgrey" />
</p>

<p align="center">
  Built with the tools and technologies:<br/>
  <img src="https://img.shields.io/badge/Markdown-black?logo=markdown" />
  <img src="https://img.shields.io/badge/Python-blue?logo=python" />
  <img src="https://img.shields.io/badge/YAML-red?logo=yaml" />
</p>

---

## 📑 Table of Contents
- [Overview](#overview)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Testing](#testing)
- [Project Structure](#project-structure)
- [License](#license)

---

## 📘 Overview

**unet3d-medical** is a comprehensive toolkit designed to streamline **3D medical image segmentation** workflows,  
from preprocessing to training and evaluation — all using a customizable **3D U-Net architecture**.

It provides a complete pipeline applicable to **multiple organs** (brain, liver, lungs, pancreas, etc.)  
and imaging modalities such as **MRI** and **CT scans**.

---

### Why use `unet3d-medical`?

This project enables researchers and developers to preprocess, train, and evaluate deep learning models  
for segmentation tasks using an **end-to-end modular architecture**.

Key features include:

- 🚀 **Pipeline Automation**: Handles resampling, patch extraction, dataset splitting, training, and evaluation.
- 🧠 **Model Architecture**: 3D U-Net with dropout, batch norm, and skip connections for volumetric data.
- 🖼️ **Interactive Visualization**: Easily inspect MRI patches, masks, and predictions during experimentation.
- 🧩 **Modular & Configurable**: Fully customizable via YAML config files and extensible modules.
- 🔁 **End-to-End Integration**: From raw NIfTI scans to trained model checkpoints and evaluation reports.

> The pipeline is organ-agnostic. Simply replace the dataset in the config and you're ready for another segmentation task.


## 🚀 Getting Started

### Prerequisites
This project requires:
- **Programming Language**: Python 3.9+
- **Package Manager**: Conda (recommended) or pip

### Installation
 **Clone the repository**
git clone https://github.com/nouamanean/pancreas-unet3d.git
cd pancreas-unet3d


