"""
shap_and_pdp.py

Generates SHAP (SHapley Additive exPlanations) and Partial Dependence Plots (PDPs)
for the tuned XGBoost model to enhance interpretability.

Outputs:
- SHAP waterfall plots for high, moderate, and low-risk individuals
- PDPs for top numeric features (e.g., age, bmi, avg_glucose_level)
"""

from cleaning.data_preprocess import preprocess_data, load_clean_data
import joblib
import pandas as pd
import shap 

def load_model_and_data():
    model = joblib.load("../../models/xgboost_tuned.pkl")
    X_train, X_test, y_train, y_test = preprocess_data()
    original_df = load_clean_data()

    # Convert all columns to float64 safely
    X_test_float = X_test.copy()
    for col in X_test_float.columns:
        try:
            X_test_float[col] = pd.to_numeric(X_test_float[col], errors='raise').astype('float64')
        except Exception as e:
            print(f"❌ Conversion failed for column '{col}': {e}")

    return model, X_test, X_test_float, y_test, original_df

def main():
    model, X_test, X_test_float, y_test, original_df = load_model_and_data()

    print("\n✅ Loaded model and data.")
    print("\n🔎 Data type check (X_test_float):")
    print(X_test_float.dtypes)

    print("\n🧪 Null check (X_test_float):")
    print(X_test_float.isnull().sum())

    print("\n🔍 Object columns (should be empty):")
    print([col for col in X_test_float.columns if X_test_float[col].dtype == 'object'])

    print(f"\n📐 Data shape: {X_test_float.shape}")

    # Now test SHAP
    print("\n🔍 Initializing SHAP Explainer...")
    explainer = shap.Explainer(model, X_test_float)
    shap_values = explainer(X_test_float)
    print("✅ SHAP values computed successfully.")

if __name__ == "__main__":
    main()
