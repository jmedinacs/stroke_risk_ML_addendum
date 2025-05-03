# Stroke Risk Prediction – Technical Summary

## Tools & Skills Demonstrated

**Languages & Libraries:**  
- Python (`pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `XGBoost`, `imbalanced-learn`, `SHAP`)  
- Google Sheets (EDA planning and logs)  
- Git/GitHub for version control  

**Techniques & Tests:**  
- Feature Engineering and Data Imputation  
- Class Imbalance Handling with SMOTE  
- Statistical Feature Selection:  
  - Chi-Square Test (categorical vs. binary target)  
  - Point-Biserial Correlation (continuous vs. binary target)  
- Machine Learning Algorithms:  
  - Logistic Regression  
  - Random Forest  
  - XGBoost (with hyperparameter tuning and threshold adjustment)  
  - K-Nearest Neighbors (with normalization)  
- Model Evaluation: Confusion Matrix, Precision, Recall, F1, ROC AUC  
- Model Explainability: SHAP Summary + Waterfall Plots, PDPs (Partial Dependence Plots)  

---

## Problem Statement
Build a predictive model to estimate stroke risk using patient health and demographic data, enabling early detection and preventative care.

## Dataset
- Source: [Kaggle – Stroke Prediction Dataset](https://www.kaggle.com/fedesoriano/stroke-prediction-dataset)  
- Rows: 5,110 patients  
- Target variable: `stroke` (0 = no, 1 = yes)  
- Class imbalance: Only ~5% of rows labeled as stroke  

---

## Data Preparation
- Imputed 201 missing `bmi` values using the median  
- Removed rare gender category 'Other'  
- Standardized and trimmed all categorical text fields  
- Applied one-hot encoding (`drop_first=True`) to nominal features  
- Applied SMOTE to oversample minority class in the training set  

## Model Training
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

## Modeling Addendum (May 2025)

This extension builds on the original stroke risk analysis by training and comparing four machine learning models:
- Logistic Regression  
- Random Forest  
- K-Nearest Neighbors (KNN)  
- XGBoost (tuned with threshold optimization)  

Key highlights:
- **XGBoost** achieved the highest recall (0.94) and AUC (0.81) after tuning.  
- SHAP and PDPs were used to explain model decisions and identify high-risk patient profiles.  
- Visual insights are saved to `/select_viz/` and discussed in the final report (coming soon).  

✅ Tuned models and evaluation scripts are located in `/src/modeling/`

---

## Model Comparison

The following leaderboard compares the performance of four machine learning models on the stroke prediction task using the same test set and evaluation metrics:

![Model Evaluation Leaderboard](select_viz/model_comparison_chart.png)

- **XGBoost** and **Logistic Regression** both achieved the highest recall of `0.48`, but XGBoost had a slightly higher F1 score and ROC AUC.  
- **KNN** and **Random Forest** underperformed, particularly in recall — a critical metric for identifying high-risk stroke patients.  
- Based on this comparison, **XGBoost** was selected for further threshold tuning and interpretability analysis using SHAP and PDP.  

📌 *Note: All models were trained on the same preprocessed dataset and evaluated on the same test set.*  

---

### Final Model: Tuned XGBoost Classifier

This version prioritized **recall** by adjusting class threshold after `RandomizedSearchCV` tuning. It's the most appropriate model for clinical use where missing stroke cases is costly.

![Confusion Matrix – Tuned XGBoost](select_viz/confusion_matrix_xgboost_tuned.png)

- **Recall**: 94%  
- **Precision**: 8%  
- **F1 Score**: 14.8%  
- **ROC AUC**: 80.7%  

> Detected **47 out of 50** stroke cases on test set.

---

## Model Interpretability

### SHAP Summary:
![SHAP Summary](select_viz/shap_feature_importance.png)
- Top positive predictors: `age`, `ever_married_yes`, `work_type_private`  
- Features like `heart_disease` had less impact than expected due to age confounding  

### SHAP Waterfall:
- Visualized high-risk, moderate-risk, and low-risk cases  
- Clear demonstration of feature contributions at the individual level  

![High Risk](select_viz/shap_high_risk.png) ![Moderate Risk](select_viz/shap_moderate_risk.png) ![Low Risk](select_viz/shap_low_risk.png)

### PDP (Partial Dependence Plots):
- `Age`: Sharp increase in stroke risk starting in mid-40s  
- `Glucose`: Spike in risk at low levels, flat afterward  
- `BMI`: Risk increases above BMI 23, plateaus around 30–40  

![PDP Age](select_viz/pdp_age.png)

---

## Project Architecture (Modularized)
- `data_preprocess.py`: Cleaning, encoding, SMOTE, scaling  
- `train_models.py`: Trains and saves all four models  
- `evaluate_models.py`: Loads models, evaluates, generates visuals  
- `shap_and_pdp.py`: Model interpretability via SHAP + PDPs  
- `model_driver.py`: Central controller to run training or evaluation  
- `main.py`: Executes the entire pipeline end-to-end  

---

## How to Use This Repository

### Requirements
Ensure the following are installed:
- Python 3.9+
- `scikit-learn`, `pandas`, `xgboost`, `matplotlib`, `shap`, `imbalanced-learn`

Install dependencies:
```bash
pip install -r requirements.txt
```

### Run the Full Pipeline
From the project root:
```bash
python pipeline/main.py
```
This runs:
1. Data cleaning
2. Preprocessing (encoding, SMOTE, scaling)
3. Model training and saving
4. Model evaluation with saved visuals
5. SHAP + PDP interpretability

Cleaned data is saved to `/data/processed/` and model outputs to `/outputs/` and `/select_viz/`

---

## Future Work: Precision-Recall Ensemble
Given the tuned XGBoost model’s high recall but low precision, a potential next step is to stage a second model as a “precision gate.”

In this two-stage ensemble, all positive predictions from the recall-focused model would be reviewed by a secondary classifier tuned for **precision**. This would reduce false positives without compromising detection of actual stroke cases — a common practice in high-stakes medical systems.

---

**Final Report and modeling log available soon.**  
[Modeling Log (Google Sheets)](https://docs.google.com/spreadsheets/d/1pduhjQ3n5z88igfg-g8DmshraBieVE_CXnfD5TDrHlg/edit#gid=1555003253)
