# Stroke Risk ML Addendum

## Overview
This project is a Machine Learning extension of the original stroke risk data analysis project. It applies machine learning models and principles to extend the findings beyond exploratory and bivariate analysis, focusing on predictive modeling to identify key stroke risk factors.

## Objectives
- Apply supervised machine learning techniques to predict stroke occurrence.
- Explore feature selection methods to identify important predictors.
- Address class imbalance through resampling techniques.
- Evaluate model performance using ROC-AUC, precision, recall, and other metrics.

## Cleaning Progress ✅
The raw dataset was inspected and cleaned using a modular Python pipeline. Key steps included:
- **BMI imputation** using the median to handle missing values in a skewed distribution.
- **Rare category removal** (gender = 'Other') due to extremely low frequency (1 occurrence).
- **Standardization of categorical text fields** (lowercased and trimmed for consistency).
- **Duplicates Removal** to remove exact duplicate records, if any.
- Cleaning progress and decisions are fully documented in a Google Sheets cleaning log.

> ✅ Dataset is now clean and prepared for Exploratory Data Analysis (EDA) and modeling.

## Status
🚧 PROJECT IN PROGRESS  
🧹 Data Cleaning Phase: **Completed**  
📊 EDA Phase: **Begins next**
