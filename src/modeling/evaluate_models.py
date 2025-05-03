"""
evaluate_models.py

Loads all trained models and compares their performance on the same test dataset.

Metrics compared:
- Precision
- Recall
- F1 Score
- ROC AUC

This script outputs a leaderboard for model selection.
"""

import joblib
import pandas as pd
import os 

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from cleaning.data_preprocess import preprocess_data



def evaluate_model(name, model, X_test, y_test):
    """
    Generates predictions and calculates performance metrics.

    Args:
        name (str): Name of the model.
        model: Trained classification model.
        X_test (pd.DataFrame or np.ndarray): Feature matrix for testing.
        y_test (pd.Series): True labels for testing.

    Returns:
        dict: Model name and its evaluation metrics.
    """    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]
    
    return {
        "Model" : name,
        "Precision" : precision_score(y_test, y_pred),
        "Recall" : recall_score(y_test, y_pred),
        "F1 Score" : f1_score(y_test, y_pred), 
        "ROC AUC" : roc_auc_score(y_test, y_prob)         
    }
    
def main():
    # Load preprocessed data
    X_train, X_test, y_train, y_test = preprocess_data()
    X_test
    
    # Load the models
    log_reg_model = joblib.load("../../models/logistic_regression_model.pkl")
    rf_model = joblib.load("../../models/random_forest_model.pkl")
    knn_model = joblib.load("../../models/knn_model.pkl")
    knn_scaler = joblib.load("../../models/knn_scaler.pkl")
    xgb_model = joblib.load("../../models/xgboost_model.pkl")
    
    # Normalized test set for KNN
    X_test_knn = knn_scaler.transform(X_test)
    
    # Evaluate all models
    results = [
        evaluate_model("Logistic Regression", log_reg_model, X_test, y_test),
        evaluate_model("Random Forest", rf_model,X_test, y_test),
        evaluate_model("KNN", knn_model, X_test_knn, y_test),
        evaluate_model("XGBoost", xgb_model, X_test, y_test),        
    ]
    
    # Convert to DataFrame
    leaderboard = pd.DataFrame(results).sort_values(by="Recall", ascending=False)
    print("\n🔍 Model Evaluation Leaderboard:")
    print(leaderboard.to_string(index=False))

if __name__ == "__main__":
    main()    
    
    
    
    