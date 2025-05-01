# Stroke Risk ML Addendum

## Overview
This project is a Machine Learning extension of the original stroke risk data analysis project. It applies machine learning models and principles to extend the findings beyond exploratory and bivariate analysis, focusing on predictive modeling to identify key stroke risk factors.

## Objectives
- Apply supervised machine learning techniques to predict stroke occurrence.
- Explore feature selection methods to identify important predictors.
- Address class imbalance through resampling techniques.
- Evaluate model performance using ROC-AUC, precision, recall, and other metrics.

## Cleaning Progress 
The raw dataset was inspected and cleaned using a modular Python pipeline. Key steps included:
- **BMI imputation** using the median to handle missing values in a skewed distribution.
- **Rare category removal** (gender = 'Other') due to extremely low frequency (1 occurrence).
- **Standardization of categorical text fields** (lowercased and trimmed for consistency).
- **Duplicates Removal** to remove exact duplicate records, if any.
- Cleaning progress and decisions are fully documented in a Google Sheets cleaning log.

> Dataset is now clean and prepared for Exploratory Data Analysis (EDA) and modeling.


## 🧠 Modeling Workflow

1. **Feature Selection**
   - Dropped `gender`, `Residence_type`, and `ID` based on chi-square and reasoning.
   - Retained `heart_disease`, `hypertension`, `ever_married`, `work_type`, `smoking_status`, and numerical features.

2. **Encoding**
   - Applied one-hot encoding with `drop_first=True` to avoid multicollinearity.

3. **Train/Test Split**
   - Stratified split (80/20) to maintain class imbalance in both sets.

4. **Resampling**
   - Applied SMOTE to training set only to correct imbalance (~5% stroke cases).
   - Verified class distribution after SMOTE.

5. **Modeling**
   - Trained Logistic Regression as baseline.
   - Evaluated using Accuracy, Recall, Precision, F1, and ROC AUC.



## Status
ROJECT IN PROGRESS  
Data Cleaning Phase: **Completed**  
EDA Phase: **Begins next**
