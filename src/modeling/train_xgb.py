"""
train_xgb.py

Trains a baseline XGBoost model for stroke prediction.

This script loads preprocessed data, trains an XGBoost classifier using default parameters,
evaluates its performance using classification metrics and ROC AUC, and saves the model
and confusion matrix for comparison with other models.
"""



from cleaning.data_preprocess import preprocess_data
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from xgboost import XGBClassifier

import matplotlib.pyplot as plt
import joblib
import os



def train_model(X_train, y_train, X_test, y_test):
    """
    Trains a baseline XGBoost classifier and evaluates its performance.

    Args:
        X_train (pd.DataFrame): Feature matrix for training.
        y_train (pd.Series): Training target labels.
        X_test (pd.DataFrame): Feature matrix for testing.
        y_test (pd.Series): True labels for testing.

    Returns:
        Tuple[XGBClassifier, np.ndarray]: Trained XGBoost model and predicted labels.
    """
    xgb_model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        eval_metric='logloss',
        random_state=42
    )
    
    xgb_model.fit(X_train, y_train)
    
    y_pred = xgb_model.predict(X_test)
    y_prob = xgb_model.predict_proba(X_test)[:,1]
    
    print("\nClassification Report (XGBoost):")
    print(classification_report(y_test, y_pred, digits=3))
    
    xgb_auc = roc_auc_score(y_test, y_prob)
    print(f"ROC AUC: {xgb_auc:.3f}")
    
    return xgb_model, y_pred
    
def create_confusion_matrix(y_test, y_pred):
    """
    Generates, displays, and saves the confusion matrix for the XGBoost model.

    Args:
        y_test (pd.Series): True stroke labels from test set.
        y_pred (np.ndarray): Predicted stroke labels from the model.

    Saves:
        Confusion matrix plot to /outputs/figures/confusion_matrix_xgb.png
    """
   
    cm = confusion_matrix(y_test, y_pred)
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Stroke","Stroke"])
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix - XGBoost")
    plt.tight_layout()
    
    
    # Save the figure
    os.makedirs("../../outputs/figures", exist_ok=True)
    plt.savefig("../../outputs/figures/confusion_matrix_xgb.png", dpi=300)
    plt.show()
    print("Confusion matrix saved to /outputs/figures/")

def train_xgboost_model():
    """
    Orchestrates baseline XGBoost model training and evaluation.

    Steps:
    - Loads preprocessed data
    - Trains XGBoost classifier with default parameters
    - Evaluates performance (classification report + ROC AUC)
    - Displays and saves confusion matrix
    - Saves the trained model for future use

    Returns:
        None
    """
    X_train, X_test, y_train, y_test = preprocess_data()
    
    xgb_model, y_pred = train_model(X_train, y_train, X_test, y_test)
    
    create_confusion_matrix(y_test, y_pred)
    
    # Check if the directory exists
    os.makedirs("../../models", exist_ok=True)    
    joblib.dump(xgb_model, "../../models/xgboost_model.pkl")
    print("XGBoost model saved.")
    
    
if __name__ == '__main__':
    train_xgboost_model() 
