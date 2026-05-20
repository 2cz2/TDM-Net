Markdown
# TDM-Net

This repository is the official PyTorch implementation of the paper:  
**"Research on Temporal Multi-modal Joint Modeling of Chest X-rays and Reports"**

---

## 📌 Introduction
TDM-Net is a temporal-aware and diagnosis-guided multi-modal network designed for longitudinal radiology report generation. By integrating historical images and prior reports via hierarchical temporal modeling, it captures disease progression dependencies over time to enhance the clinical utility of generated text.

---

## 🛠️ Environmental Setup
```bash
git clone [https://github.com/2cz2/TDM-Net.git](https://github.com/2cz2/TDM-Net.git)
cd TDM-Net
pip install -r requirements.txt
📊 Data Preparation
The MIMIC-CXR-JPG dataset is available at PhysioNet. Please organize your dataset directory as follows:

Plaintext
data/
└── mimic-cxr/
    ├── images/           # Raw DICOM or JPEG images
    ├── reports/          # Original radiology report text files
    └── annotation.json   # Metadata containing temporal splits and labels
💡 Acknowledge: The data preprocessing and dataset split are based on the implementation from CelestialShine/Longitudinal-Chest-X-Ray.

🚀 Train the Model
Run the following command to train the model on the MIMIC-CXR data:

Bash
bash run_mimic_cxr.sh
