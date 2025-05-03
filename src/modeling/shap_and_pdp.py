"""
shap_and_pdp.py

Generates SHAP (SHapley Additive exPlanations) and Partial Dependence Plots (PDPs)
for the tuned XGBoost model to enhance interpretability.

Outputs:
- SHAP waterfall plots for high, moderate, and low-risk individuals
- PDPs for top numeric features (e.g., age, bmi, avg_glucose_level)
"""

from cleaning.data_preprocess import preprocess_data, load_clean_data
from sklearn.inspection import PartialDependenceDisplay

import joblib
import pandas as pd
import shap 
import matplotlib.pyplot as plt 
import os 

def load_model_and_data():
    model = joblib.load("../../models/xgboost_tuned.pkl")
    X_train, X_test, y_train, y_test = preprocess_data()
    original_df = load_clean_data()

    # Ensure all SHAP features are numeric (required)
    X_test_float = X_test.copy()
    for col in X_test_float.columns:
        try:
            X_test_float[col] = pd.to_numeric(X_test_float[col], errors='raise').astype('float64')
        except Exception as e:
            print(f"Conversion failed for column '{col}': {e}")

    return model, X_test, X_test_float, y_test, original_df

def generate_shap_summary(shap_values):
    """
    Generates and saves a SHAP summary bar plot showing global feature importance.

    Args:
        shap_values (shap.Explanation): Computed SHAP values for the dataset.
    """
    os.makedirs("../../select_viz", exist_ok = True)
    
    # Plot Shap Summary
    plt.figure()
    shap.plots.bar(shap_values, show=False)
    plt.title("SHAP Feature Importance – Mean Absolute Impact")
    plt.tight_layout()
    plt.savefig("../../select_viz/shap_feature_importance.png", dpi=300)
    plt.show()
    plt.close()
    
def generate_shap_waterfall(shap_values, row_idx, label="Example"):
    """
    Generates and saves a SHAP waterfall plot for a specific prediction.

    Args:
        shap_values (shap.Explanation): SHAP values object.
        row_idx (int): Index of the row to visualize.
        label (str): Custom label for title and filename (e.g., 'High', 'Low', 'Moderate').

    Saves:
        Waterfall plot to /select_viz/shap_<label>_risk.png
    """
    os.makedirs("../../select_viz", exist_ok=True)
    
    plt.figure()
    shap.plots.waterfall(shap_values[row_idx], show = False)
    plt.title(f"SHAP – {label} Risk Example")
    plt.tight_layout()
    plt.savefig(f"../../select_viz/shap_{label.lower()}_risk.png", dpi=300)
    plt.show()
    plt.close()
    print(f"SHAP waterfall plot saved for {label} risk.")

def generate_waterfall_for_risk_levels(model, X_test, X_test_float, y_test, original_df, shap_values):
    """
    Identifies high, moderate, and low risk patients and generates SHAP waterfall plots.

    Args:
        model: Trained model.
        X_test: Original test features (unconverted).
        X_test_float: Float-converted test data (for SHAP).
        y_test: True labels.
        original_df: Cleaned full dataset (for metadata).
        shap_values: SHAP Explanation object.
    """
    
    # Predict stroke probabilities
    probs = model.predict_proba(X_test_float)[:, 1]

    # Merge test features with predicted probabilities and metadata
    results = X_test.copy()
    results["prob_stroke"] = probs
    results["true_stroke"] = y_test.values
    results["heart_disease"] = original_df.loc[X_test.index, "heart_disease"].values

    # --- HIGH RISK (with heart disease, highest prob) ---
    high_risk = results[results["heart_disease"] == 1].sort_values("prob_stroke", ascending=False)
    high_idx = X_test.index.get_loc(high_risk.index[0])
    generate_shap_waterfall(shap_values, row_idx=high_idx, label="High")

    # --- LOW RISK (lowest prob overall) ---
    low_risk = results.sort_values("prob_stroke", ascending=True)
    low_idx = X_test.index.get_loc(low_risk.index[0])
    generate_shap_waterfall(shap_values, row_idx=low_idx, label="Low")

    # --- MODERATE RISK (probability between 0.3 and 0.5) ---
    moderate_risk = results[(results["prob_stroke"] >= 0.3) & (results["prob_stroke"] <= 0.5)]
    if not moderate_risk.empty:
        mod_idx = X_test.index.get_loc(moderate_risk.index[0])
        generate_shap_waterfall(shap_values, row_idx=mod_idx, label="Moderate")

def plot_PDP(model, X_test_float, feature_name):
    """
    Generates and saves a Partial Dependence Plot (PDP) for a given numeric feature.

    Args:
        model: Trained classifier (e.g., tuned XGBoost).
        X_test_float (pd.DataFrame): Float-converted test data.
        feature_name (str): The name of the numeric feature to plot.

    Saves:
        A PNG image of the PDP to /select_viz/pdp_<feature_name>.png
    """
    
    fig, ax = plt.subplots(figsize=(8,6))
    PartialDependenceDisplay.from_estimator(
        model,
        X_test_float,
        features = [feature_name],
        kind = "average",
        grid_resolution = 100,
        ax = ax
    )
    plt.title(f"Partial Dependence Plot – {feature_name} vs Stroke Probability")
    plt.tight_layout()
    plt.savefig(f"../../select_viz/pdp_{feature_name}.png", dpi=300)
    plt.show()
    plt.close()
    



def main():
    model, X_test, X_test_float, y_test, original_df = load_model_and_data()

    # Confirm input integrity
    print("\nLoaded model and data.")
    print("\nData type check (X_test_float):")
    print(X_test_float.dtypes)

    print("\nNull check (X_test_float):")
    print(X_test_float.isnull().sum())

    print("\nObject columns (should be empty):")
    print([col for col in X_test_float.columns if X_test_float[col].dtype == 'object'])

    print(f"\nData shape: {X_test_float.shape}")

    # Run SHAP
    print("\nInitializing SHAP Explainer...")
    explainer = shap.Explainer(model, X_test_float)
    shap_values = explainer(X_test_float)
    print("SHAP values computed successfully.")
    
    # Visual outputs
    generate_shap_summary(shap_values)
    generate_waterfall_for_risk_levels(model, X_test, X_test_float, y_test, original_df, shap_values)

    # PDPs for top numeric features
    plot_PDP(model, X_test_float, "age")
    plot_PDP(model, X_test_float, "bmi")
    plot_PDP(model, X_test_float, "avg_glucose_level")
   

if __name__ == "__main__":
    main()
