Markdown
# TDM-Net: Research on Temporal Multi-modal Joint Modeling of Chest X-rays and Reports

## 📌 Introduction
In this study, we propose a novel **Temporal-aware and Diagnosis-guided Multi-modal Network (TDM-Net)**. By integrating historical images and prior reports, the framework utilizes a hierarchical temporal structure to capture cross-time-step disease progression dependencies, and introduces a diagnosis-guided mechanism to reformulate disease classification results into clinical prompt vectors that adaptively guide the decoding process. Experimental results on the MIMIC-CXR dataset demonstrate that our method significantly improves clinical entity recall and Macro-F1 scores, effectively enhancing the clinical utility of generated reports.

## 🛠️ Environmental Setup
```bash
git clone [https://github.com/2cz2/TDM-Net.git](https://github.com/2cz2/TDM-Net.git)
cd TDM-Net
pip install -r requirements.txt
📊 Data Preparation
The MIMIC-CXR-JPG dataset is available at: PhysioNet.

Please organize your MIMIC-CXR dataset directory as follows to support longitudinal data loading:

Plaintext
data/
└── mimic-cxr/
    ├── images/           # Raw DICOM or JPEG images
    ├── reports/          # Original radiology report text files
    └── annotation.json   # Metadata containing temporal splits and labels
💡 Acknowledge: The data preprocessing pipeline and dataset split in this project are based on the implementation from the baseline repository: CelestialShine/Longitudinal-Chest-X-Ray. We sincerely thank the authors for their open-source contribution.

🚀 Train the model
Run the following command to train a model on the MIMIC-CXR data:

Bash
bash run_mimic_cxr.sh
