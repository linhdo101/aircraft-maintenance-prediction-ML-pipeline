# Aircraft Maintenance Prediction

### Applied Machine Learning for Predictive Aircraft Maintenance

**CS610 Applied Machine Learning**

A machine-learning pipeline for predicting near-term aircraft maintenance risk from multivariate flight sensor data.

The project compares classical machine learning, time-series feature extraction, deep learning, and ensemble methods using flight data derived from the **National General Aviation Flight Information Database (NGAFID)**.

---

## Project Overview

The objective is to predict whether an aircraft flight occurs within **2 days of an unplanned maintenance event**.

Each flight is represented using multivariate sensor measurements collected over time. The project explores two complementary modeling approaches:

* **Tabular feature engineering** for XGBoost
* **Raw time-series modeling** using MiniRocket, InceptionTime, and a Transformer

The final stage combines the individual models using multiple ensemble strategies.

---

## Models

The following approaches were developed and evaluated:

| Model             | Approach                                                        |
| ----------------- | --------------------------------------------------------------- |
| **XGBoost**       | Gradient-boosted trees on aggregated flight-level features      |
| **MiniRocket**    | Random convolutional kernels for time-series feature extraction |
| **InceptionTime** | Multi-scale 1D convolutional neural network                     |
| **Transformer**   | Attention-based neural network for multivariate time series     |
| **Ensemble**      | Soft voting, weighted voting, and logistic stacking             |

---

## Data Pipeline

```text
Raw NGAFID-derived flight data
            ↓
    Exploratory analysis
            ↓
    Data preprocessing
            ↓
 ┌──────────┴──────────┐
 │                     │
Tabular features    Time-series sequences
 │                     │
XGBoost        ┌────────┼───────────┐
               │        │           │
          MiniRocket InceptionTime Transformer
               │        │           │
               └────────┼───────────┘
                        ↓
                    Ensemble
```

The preprocessing pipeline produces:

* flight-level labels and train/validation/test splits,
* aggregated tabular features,
* standardized multivariate time-series sequences,
* master indices for aligning sequences with labels.

The time-series representation contains **19,540 flights**, each represented with up to **9,212 timesteps and 15 sensor channels**.

---

## Target

The primary target is:

```text
rul_2d
```

where:

```text
0 = at-risk flight
1 = safe flight
```

An **at-risk** flight is one occurring within two days of an unplanned maintenance event.

Because failing to identify an at-risk aircraft is more costly than generating a false alarm, the evaluation places particular emphasis on **recall for the at-risk class** in addition to ROC-AUC and PR-AUC.

---

## Final Results

Performance on the shared held-out test set:

| Model                 |   ROC-AUC |    PR-AUC | At-Risk Recall | At-Risk F1 |
| --------------------- | --------: | --------: | -------------: | ---------: |
| **Stacking Ensemble** | **0.778** | **0.670** |          0.838 |  **0.691** |
| Weighted Voting       |     0.775 |     0.669 |          0.846 |      0.688 |
| Soft Voting           |     0.775 |     0.669 |          0.848 |      0.687 |
| InceptionTime         |     0.762 |     0.643 |          0.867 |          — |
| Transformer           |     0.754 |     0.640 |          0.827 |          — |
| MiniRocket            |     0.740 |     0.634 |          0.869 |          — |
| XGBoost               |     0.725 |     0.639 |          0.820 |          — |

The **stacking ensemble** achieved the highest overall ROC-AUC.

However, bootstrap confidence intervals overlap with the best individual model, InceptionTime, so the observed ensemble improvement should not be interpreted as statistically conclusive.

---

## Ensemble Strategy

Three ensemble approaches were evaluated:

### Soft Voting

Simple average of predicted probabilities across all models.

### Weighted Voting

Model probabilities are weighted according to validation ROC-AUC performance.

### Stacking

A logistic-regression meta-model is trained on the validation-set predictions from:

* XGBoost
* MiniRocket
* InceptionTime
* Transformer

and then evaluated on the held-out test set.

This avoids training the meta-model directly on test predictions.

---

## Repository Structure

```text
aircraft-maintenance-prediction/
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Data_Preprocessing.ipynb
│   ├── 03_XGB.ipynb
│   ├── 04_MiniRocket_Features_Extract.ipynb
│   ├── 05_MiniRocket.ipynb
│   ├── 06_InceptionTime.ipynb
│   ├── 07_Transformer.ipynb
│   └── 08_Ensemble.ipynb
│
├── outputs/
│   ├── labels.csv
│   ├── seq_indices.npy
│   ├── XGB.xls
│   ├── MiniRocket.csv
│   ├── InceptionTime.csv
│   ├── Transformer.csv
│   └── Ensemble.csv
│
└── docs/
    └── USER_GUIDE.md
```

---

## Notebook Workflow

### `01_EDA.ipynb`

Exploratory analysis of the raw aviation sensor data and maintenance structure.

### `02_Data_Preprocessing.ipynb`

Builds the common modeling dataset, including:

* labels,
* train/validation/test splits,
* tabular features,
* standardized time-series sequences,
* sequence-to-label alignment indices.

### `03_XGB.ipynb`

Trains an XGBoost model using aggregated flight-level features.

### `04_MiniRocket_Features_Extract.ipynb`

Transforms the multivariate time-series data using MiniRocket kernels.

Because the resulting feature matrices are very large, they are not stored in this repository and should be regenerated before running the MiniRocket model.

### `05_MiniRocket.ipynb`

Trains and evaluates models using MiniRocket-derived time-series features.

### `06_InceptionTime.ipynb`

Implements an InceptionTime architecture with multi-scale one-dimensional convolutions and residual connections.

### `07_Transformer.ipynb`

Implements a PyTorch Transformer encoder for multivariate flight sensor sequences, including masking for variable flight lengths and temporal padding.

### `08_Ensemble.ipynb`

Compares all individual models and builds:

* soft-voting ensemble,
* validation-weighted ensemble,
* logistic stacking ensemble,
* bootstrap confidence intervals,
* calibration analysis.

---

## Data

The original dataset was obtained from Kaggle:

**Aviation Maintenance Dataset from the NGAFID**

https://www.kaggle.com/datasets/hooong/aviation-maintenance-dataset-from-the-ngafid

The dataset is derived from the **National General Aviation Flight Information Database (NGAFID)**.

### Large files not included

Several large artifacts are intentionally excluded from this repository:

```text
2days.tar.gz
sequences.npy
tabular_features_scaled.csv
MiniRocket transformed feature matrices
```

These files are omitted because of their size.

They can be reconstructed by downloading the original dataset and executing the preprocessing pipeline.

The repository does include:

```text
labels.csv
seq_indices.npy
```

These provide the target labels, dataset splits, and sequence-to-flight alignment used throughout the modeling pipeline.

---

## Reproducing the Project

### 1. Download the dataset

Download the **Aviation Maintenance Dataset from the NGAFID** from Kaggle.

### 2. Run the exploratory analysis

```text
01_EDA.ipynb
```

### 3. Run preprocessing

```text
02_Data_Preprocessing.ipynb
```

This recreates the derived modeling artifacts, including:

```text
labels.csv
seq_indices.npy
sequences.npy
tabular_features.csv
tabular_features_scaled.csv
```

### 4. Train the individual models

Run:

```text
03_XGB.ipynb
04_MiniRocket_Features_Extract.ipynb
05_MiniRocket.ipynb
06_InceptionTime.ipynb
07_Transformer.ipynb
```

### 5. Build the final ensemble

After generating the individual model predictions, run:

```text
08_Ensemble.ipynb
```

---

## Development Environment

The project was developed collaboratively using **Kaggle notebooks** with GPU acceleration.

Some notebooks retain Kaggle-specific filesystem paths such as:

```text
/kaggle/input/
/kaggle/working/
```

To run the project locally, update these paths to match your local directory structure.

---

## Team

**Group 10 — CS610 Applied Machine Learning**

* Kimberly Bertoli
* Do Thi Phuong Linh
* Nguyen Kim Hau
* Tanya Arora
* Cheung See Lin

---

## Key Takeaway

The project demonstrates that combining heterogeneous models can improve predictive performance for aircraft maintenance risk.

The strongest individual model was **InceptionTime**, while the **stacking ensemble** achieved the highest overall ROC-AUC. At the same time, bootstrap analysis showed overlapping confidence intervals, highlighting the importance of evaluating whether apparent ensemble gains are statistically meaningful rather than relying only on point estimates.

