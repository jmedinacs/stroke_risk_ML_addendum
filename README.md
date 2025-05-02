## Stroke Risk Prediction – Technical Summary

### 🔍 Problem Statement
Build a predictive model to estimate stroke risk using patient health and demographic data, enabling early detection and preventative care.

### 📦 Dataset
- Source: [Kaggle – Stroke Prediction Dataset](https://www.kaggle.com/fedesoriano/stroke-prediction-dataset)
- Rows: 5,110 patients
- Target variable: `stroke` (0 = no, 1 = yes)
- Class imbalance: Only ~5% of rows labeled as stroke

---

### 🧹 Data Preparation
- Imputed 201 missing `bmi` values using the median
- Removed rare gender category 'Other'
- Standardized and trimmed all categorical text fields
- Applied one-hot encoding (`drop_first=True`) to nominal features
- Applied SMOTE to oversample minority class in the training set

### 🧪 Model Training
Trained and evaluated four models:
- **Logistic Regression** (baseline)
- **Random Forest**
- **XGBoost** (baseline and tuned)
- **K-Nearest Neighbors** (with normalization)

All models were trained using an **80/20 stratified split** and evaluated on:
- Recall (stroke = 1)
- Precision (stroke = 1)
- F1 Score
- ROC AUC
- Confusion Matrix

---

### 🥈 Original Champion Model: XGBoost (Baseline)
- Balanced model performance across metrics
- Good general-purpose classifier

**Metrics:**
- Recall: 48%
- Precision: 18.8%
- F1 Score: 27%
- ROC AUC: 80.6%

Saved as `xgboost_model.pkl`

---

### 🥇 Final Champion Model: XGBoost (Recall-Optimized)
- Tuned hyperparameters via `RandomizedSearchCV`
- Focused on maximizing recall for stroke cases
- Adjusted threshold to 0.5 for optimal detection

**Metrics:**
- Recall: **94%**
- Precision: 8%
- F1 Score: 14.8%
- ROC AUC: 80.7%

Saved as `xgboost_model_tuned.pkl`

> This version detected **47 out of 50** stroke cases and is ideal for clinical scenarios where false negatives are unacceptable.

---

### 🧠 Model Interpretability
#### SHAP Summary:
- Top positive predictors: `age`, `ever_married_yes`, `work_type_private`
- Features like `heart_disease` had less impact than expected due to age confounding

#### SHAP Waterfall:
- Visualized high-risk, moderate-risk, and low-risk cases
- Clear demonstration of feature contributions at the individual level

#### PDP (Partial Dependence Plots):
- `Age`: Sharp increase in stroke risk starting in mid-40s
- `Glucose`: Spike in risk at low levels, flat afterward
- `BMI`: Risk increases above BMI 23, plateaus around 30–40

---

### 🧱 Project Architecture (Planned Modularization)
- `data_preprocessor.py`: Cleaning, encoding, SMOTE, scaling
- `train_models.py`: Trains and saves all four models
- `evaluate_models.py`: Loads models, evaluates, generates visuals
- `model_driver.py`: Central controller to run training or evaluation

---

## ⚠️ UNDER CONSTRUCTION
Please excuse the messy `model_training.py` file. This fully functional modeling pipeline (Logistic Regression, Random Forest, XGBoost, and KNN) is currently being modularized for clarity and maintainability.

The cleaned, modular version will include:
- Isolated data preprocessing pipeline
- Dedicated training scripts for each model
- Central driver to manage training, evaluation, and prediction

This README and codebase will be updated soon to reflect the new structure and make exploration easier for reviewers and collaborators.

---

**Next: modularize the codebase and finalize report for launch.**

For a quick look at this projet's modeling documentation and results [view-model-log](https://docs.google.com/spreadsheets/d/1pduhjQ3n5z88igfg-g8DmshraBieVE_CXnfD5TDrHlg/edit?gid=1555003253#gid=1555003253)

