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
git clone [https://github.com/2cz2/TDM-Net.git](https://github.com/2cz2/TDM-Net.git)
cd TDM-Net
pip install -r requirements.txt
```
## 📊 Data Preparation
The MIMIC-CXR-JPG dataset is available at PhysioNet. Please organize your dataset directory as follows:

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
## 🚀 Train the Model
Run the following command to train the model on the MIMIC-CXR data:

```bash
bash run_mimic_cxr.sh
```
