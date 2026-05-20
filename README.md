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
    ├── images/           
    ├── reports/          
    └── annotation.json   
💡 Acknowledge: The data preprocessing pipeline and dataset split in this project are based on the implementation from the baseline repository: https://github.com/CelestialShine/Longitudinal-Chest-X-Ray. We sincerely thank the authors for their open-source contribution.

🚀 Train the Model
Run the following command to train the model on the MIMIC-CXR data:

Bash
bash run_mimic_cxr.sh
