"""
train_all_models.py

Coordinates the training of all machine learning models used in the stroke risk prediction project:
- Logistic Regression
- Random Forest
- K-Nearest Neighbors
- XGBoost (tuned)

Each model is trained on the same preprocessed dataset and saved to disk.

Author: John Medina
Date: 2025-05-03
Project: Stroke Risk ML Addendum
"""

import modeling.train_logistic_regression as lr_model
import modeling.train_knn as knn_model
import modeling.train_xgb as xgb_model
import modeling.train_rf as rf_model
import modeling.xgb_tuning as tuned_xgb_model
from cleaning.data_preprocess import preprocess_data


def train_all_models():
    lr_model.train_logistic_regression_model()
    knn_model.train_knn_model()
    xgb_model.train_xgboost_model()
    rf_model.train_random_forest_model()
    tuned_xgb_model.tune_xgb_model()
    print("\nAll models trained and saved.")


if __name__ == "__main__":
    train_all_models()
