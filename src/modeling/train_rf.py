"""
train_random_forest.py

Trains and evaluates a Random Forest model for stroke prediction.

This module loads preprocessed data, fits a Random Forest classifier using 
the balanced training set, evaluates the model on the original test set, 
displays and saves the confusion matrix, and stores the trained model to disk.

Returns visual and numerical evaluation metrics for downstream reporting.
"""

from cleaning.data_preprocess import preprocess_data
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt 
import joblib 
import os 


def train_model(X_train, y_train, X_test, y_test):
    """
    Trains a Random Forest classifier and evaluates it on the test set.

    Args:
        X_train (pd.DataFrame): SMOTE-balanced training features.
        y_train (pd.Series): Balanced training labels.
        X_test (pd.DataFrame): Original test features (imbalanced).
        y_test (pd.Series): Original test labels.

    Returns:
        Tuple[RandomForestClassifier, np.ndarray]: Trained model and test set predictions.
    """
    rf_model = RandomForestClassifier(
        n_estimators = 100,
        random_state=42,
        class_weight='balanced'
    )
    
    rf_model.fit(X_train, y_train)
    
    y_pred = rf_model.predict(X_test)
    y_prob = rf_model.predict_proba(X_test)[:,1] 
    
    print("\nClassification Report (Random Forest):")
    print(classification_report(y_test, y_pred, digits=3))
    
    rf_auc = roc_auc_score(y_test, y_prob)
    print(f"ROC AUC: {rf_auc:.3f}")
    
    return rf_model, y_pred


def create_confusion_matrix(y_test, y_pred):
    """
    Generates, displays, and saves the confusion matrix.

    Args:
        y_test (pd.Series): True stroke labels from test set.
        y_pred (np.ndarray): Predicted stroke labels from the model.

    Saves:
        Confusion matrix plot to /outputs/figures/
    """
   
    cm = confusion_matrix(y_test, y_pred)
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Stroke","Stroke"])
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix - Random Forest")
    plt.tight_layout()
    plt.show()
    
    # Save the figure
    os.makedirs("../../outputs/figures", exist_ok=True)
    plt.savefig("../../outputs/figures/confusion_matrix_rf.png", dpi=300)
    print("Confusion matrix saved to /outputs/figures/")


def train_random_forest_model():
    """
    Orchestrates preprocessing, training, evaluation, and saving of 
    the Random Forest model for stroke classification.

    Executes the training pipeline using preprocessed data, evaluates 
    the model, saves the confusion matrix plot, and stores the trained model.
    """
    # Retrieve the training and test sets from preprocessing
    X_train, X_test, y_train, y_test = preprocess_data()
    
    # Train the model
    rf_model, y_pred = train_model(X_train, y_train, X_test, y_test)
    
    # Create and save the confusion matrix
    create_confusion_matrix(y_test, y_pred)
    
    # Check if the directory exists
    os.makedirs("../../models", exist_ok=True)
    
    # Save the model for future use.
    joblib.dump(rf_model, "../../models/random_forest_model.pkl")
    print("Random Forest Model saved to /models")
    

if __name__ == '__main__':
    train_random_forest_model()