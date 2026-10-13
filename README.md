# TDM-Net

This repository is the official PyTorch implementation of the paper:  
**"TDM-Net: A Temporal-aware and Diagnosis-guided Multi-modal Network for Longitudinal
Radiology Report Generation"**

---

## 📌 Introduction
TDM-Net is a temporal-aware and diagnosis-guided multimodal network for longitudinal chest X-ray report generation. It preserves an independent current-image pathway while integrating prior images and reports through temporal encoding, gated memory fusion, and image-predicted diagnostic prompts. This design supports report generation grounded in current observations and analysis of how historical information contributes to both correct findings and reporting errors.

---

## 🛠️ Environmental Setup
```bash
git clone https://github.com/2cz2/TDM-Net.git
cd TDM-Net
pip install -r requirements.txt
```
## 📊 Data Preparation
1. Obtain the MIMIC-CXR reports and MIMIC-CXR-JPG images, metadata, and CheXpert labels from PhysioNet. Organize the source files as follows, preserving the original image subdirectories:

```text
TDM-Net/
├── data/
│   └── mimic-cxr/
│       ├── images/
│       ├── mimic-cxr-reports/
│       ├── annotation.json
│       └── annotation_with_labels.json
├── mimic-cxr-2.0.0-metadata.csv
└── mimic-cxr-2.0.0-chexpert.csv
```
2. Check the paths in `config/paths.json`. run:

```bash
python rebuild_annotation.py
```
## 🚀 Train the Model
Download the [ImageNet ResNet-101 V1 weights](https://download.pytorch.org/models/resnet101-63fe2227.pth) to `weights/torch/hub/checkpoints/resnet101-63fe2227.pth`, then run:

```bash
python run_pipeline.py prepare --execute
bash run_mimic_cxr.sh
```

## 🙏 Acknowledgement
TDM-Net is built upon the awesome [Longitudinal-Chest-X-Ray](https://github.com/CelestialShine/Longitudinal-Chest-X-Ray). We thank the authors for sharing their code.
