"""
xgb_tuning.py

Performs hyperparameter tuning for XGBoost using RandomizedSearchCV.
After training the best model (optimized for recall), this script:
- Evaluates classification performance
- Performs threshold tuning
- Saves the tuned model for future use
"""

from cleaning.data_preprocess import preprocess_data
from sklearn.metrics import classification_report, roc_auc_score, precision_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, recall_score, f1_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.inspection import PartialDependenceDisplay
from xgboost import XGBClassifier, plot_importance

import shap
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os


def evaluate_thresholds(y_test, y_prob, thresholds=[0.5, 0.4, 0.3, 0.25, 0.2]):
    """
    Evaluates model performance across multiple probability thresholds.

    For each threshold, prints:
    - Precision
    - Recall
    - F1 Score
    - Confusion Matrix

    Args:
        y_test (pd.Series): True binary target values.
        y_prob (np.ndarray): Predicted probabilities for the positive class.
        thresholds (list): List of thresholds to evaluate.
    """
    print("\nThreshold Tuning – XGBoost")    

    for thresh in thresholds:
        y_pred_thresh = (y_prob >= thresh).astype(int)
        
        precision = precision_score(y_test, y_pred_thresh)
        recall = recall_score(y_test, y_pred_thresh)
        f1 = f1_score(y_test, y_pred_thresh)
        cm = confusion_matrix(y_test, y_pred_thresh)
        
        
        print(f"\nThreshold: {thresh}")
        print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1 Score: {f1:.3f}")
        print(f"Confusion Matrix:\n{cm}")

def tune_xgb_model():
    """
    Performs hyperparameter tuning for XGBoost using RandomizedSearchCV,
    evaluates best model, performs threshold analysis, and saves the model.
    """
    X_train, X_test, y_train, y_test = preprocess_data()
    
    # Step 1: Set up base classifier
    xgb_clf = XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder = False,
        random_state = 42
    )
    
    # Step 2: Define hyperparameter search space
    param_dist = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3,5,7],
        'subsample': [0.7, 0.8, 1.0],
        'scale_pos_weight': [1,5,10,20]
    }
    
    # Step 3: Search
    DEBUG_MODE = False 
    
    random_search = RandomizedSearchCV(
        estimator = xgb_clf,
        param_distributions = param_dist,
        n_iter = 50,
        scoring = 'recall',
        cv = 5,
        verbose = 2 if DEBUG_MODE else 0,
        n_jobs = -1,
        random_state = 42
    )
    
    random_search.fit(X_train, y_train)
    best_model = random_search.best_estimator_
    
    # Step 4: Predict and evaluate
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:,1]
    
    print("\nClassification Report – Tuned XGBoost:")
    print(classification_report(y_test, y_pred, digits=3))
    print(f"ROC AUC: {roc_auc_score(y_test, y_prob):.3f}")
    
    # Step 5: Threshold tuning
    evaluate_thresholds(y_test, y_prob)

    # Step 6: Save tuned model
    os.makedirs("../../models", exist_ok=True)
    joblib.dump(best_model, "../../models/xgboost_tuned.pkl")
    print("Tuned XGBoost model saved.")
    
    
    
    
if __name__ == '__main__':
    tune_xgb_model()