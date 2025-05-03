"""
main.py

Main execution script for the Stroke Risk ML Addendum project.

This script loads raw data, applies data cleaning steps, and saves the cleaned dataset.
Subsequent EDA and modeling steps are coordinated from this file or relevant drivers.

Author: John Medina
Date: 2025-04-30
Project: Stroke Risk ML Addendum
"""

from cleaning.data_cleaning import clean_data, inspect_data
from utils.data_io import load_raw_data
from cleaning.data_preprocess import preprocess_data
from modeling.train_all_models import train_all_models
from modeling.evaluate_models import initiate_model_evaluations
from modeling.shap_and_pdp import evaluate_shap_and_pdp


def main():
    # Step 1: Load and inspect raw data
    df_raw = load_raw_data("data/raw/stroke_data.csv")
    inspect_data(df_raw)

    # Step 2: Clean and save cleaned data
    df_clean = clean_data(df_raw)
    df_clean.to_csv("data/processed/stroke_cleaned.csv", index=False)
    print("✅ Cleaned data saved.")

    # Step 3: Preprocess (splits, encodings, SMOTE, scaling)
    X_train, X_test, y_train, y_test = preprocess_data()

    # Step 4: Train and save all models
    train_all_models()

    # Step 5: Evaluate all models (prints + saves visualizations)
    initiate_model_evaluations()

    # Step 6: SHAP + PDP interpretation for best model
    evaluate_shap_and_pdp()

if __name__ == "__main__":
    main()
