# Deep Anomaly Detection for Credit Card Fraud

This project investigates and compares anomaly detection models for detecting fraudulent transactions using a normalized credit card dataset. It combines classical and deep learning-based techniques to identify patterns that indicate fraud, especially when the data is imbalanced or sparse.

Authors: Julia Jiang and Fiona Zhang

---
## Link to our amazing GitHub Repository!

```
https://github.com/juliajiang218/DM-assignment6.git
```
---

##  What This Project Does

The pipeline includes:
- **Data understanding & preprocessing**: Generates an EDA report, splits and processes data for model training.
- **Model training**: Trains multiple anomaly detection models including:
  - `IsolationForest` (baseline)
  - `DevNet1` — standard DevNet model
  - `DevNet2` — DevNet with feature engineering
- **Evaluation**: Selects the best model based on AUC-ROC and AUC-PR scores on a held-out test set.
- **Synthetic testing**: Evaluates the best model on **synthetically generated datasets** of increasing sizes using SMOTE to simulate unseen data conditions and stress-test model generalizability.

---

## Additional Dependency Required:

```
pip install imbalanced-learn
```

---

## Directory Structure

```
FraudDetection/
├── data/
│   └── creditcardfraud_normalised.csv         # Input dataset
├── outputs/
│   ├── data_report/
│   │   └── data_report.txt                    # EDA summary (nulls, types, outliers, etc.)
│   ├── evaluation/
│   │   ├── result.txt                         # Evaluation metrics for all models
│   │   └── unseen_data_performance.txt        # Best model tested on synthetic datasets
│   ├── pipeline/
│   │   └── best_model.pkl                     # Serialized best-performing model (DevNet1)
│   ├── report.pdf                             # progress report
│   └── literature_survey.pdf                  # literature survey
├── scripts/
│   ├── assignment_6.py                        # Main pipeline script
│   ├── devnet_class.py                        # DevNet architecture class
│   ├── devnet_utils.py                        # DevNet training helpers
│   ├── devnet.py                              # DevNet wrapper
│   ├── utils.py                               # Data cleaning, transformation, and report utilities
│   └── __pycache__/                           # Python cache
└── README.md                                  # This file
```

---

## How to Run

Make sure you're in the `scripts/` directory, and run:

```bash
python assignment_6.py
```

---

## What the Reports Show

### `data_report.txt`
A data quality overview:
- Column types and resolutions
- Duplicate records
- Null counts and percentage
- Value range, mean, median, and top 10% outliers per column

### `result.txt`
Evaluation scores for all three anomaly detection models:
- `AUC-ROC`: Ability to distinguish between fraud and non-fraud
- `AUC-PR`: Precision-recall curve score, especially useful for imbalanced data

Each model's result block includes:
```
Model Name: DevNet1
AUC-ROC: 0.9543
AUC-PR:  0.8817
```

### `unseen_data_performance.txt`
Shows how the **best model (DevNet1)** performs on synthetic datasets of increasing size:

```
Test Set of Size: 1714
AUC-ROC: 0.9482
AUC-PR:  0.8601
----------------------------------------
```

This tests how well the model generalizes to *unseen distributions*, especially under heavy class imbalance or volume changes.

---

## Authors

- Julia Jiang  
- Fiona Zhang

---

## Disclaimer & Licensing

This script is provided for educational purposes. Unauthorized reproduction, distribution, or use is strictly prohibited without the consent of the authors. See the top of `assignment_6.py` for full legal terms.
