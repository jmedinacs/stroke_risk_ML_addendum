"""
train_logistic_regression.py

Trains and evaluates a Logistic Regression model for stroke prediction.

This module loads preprocessed data, fits a Logistic Regression model using the 
balanced training data, evaluates performance on the original imbalanced test set, 
displays and saves the confusion matrix, and persists the trained model to disk.

Returns visual and numerical metrics for model performance evaluation.
"""

from cleaning.data_preprocess import preprocess_data
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import classification_report, roc_auc_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt 
import joblib 
import os 


def train_model(X_train, y_train, X_test, y_test):
    """
    Trains a Logistic Regression model and evaluates it on the test set.

    Args:
        X_train (pd.DataFrame): Training feature matrix (SMOTE-applied).
        y_train (pd.Series): Balanced target vector for training.
        X_test (pd.DataFrame): Original feature matrix for testing.
        y_test (pd.Series): Original target vector for testing.

    Returns:
        Tuple[LogisticRegression, np.ndarray]: Trained model and predicted labels on test set.
    """

    # Initialize and train logistic regression model
    log_reg_model = LogisticRegression(max_iter=1000, random_state=42)
    log_reg_model.fit(X_train, y_train)
    
    y_pred = log_reg_model.predict(X_test)
    y_prob = log_reg_model.predict_proba(X_test)[:,1]
    
    # Evaluation metrics
    print("\n Classification Report (Logistic Regression): ")
    print(classification_report(y_test, y_pred, digits=3))

    auc = roc_auc_score(y_test, y_prob)
    print(f"ROC AUC: {auc: 3f}")
    
    return log_reg_model, y_pred

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
    plt.title("Confusion Matrix - Logistic Regression")
    plt.tight_layout()
    plt.show()
    
    # Save the figure
    os.makedirs("../../outputs/figures", exist_ok=True)
    plt.savefig("../../outputs/figures/confusion_matrix_logreg.png", dpi=300)
    print("Confusion matrix saved to /outputs/figures/")

def train_logistic_regression_model():
    """
    Orchestrates the preprocessing, training, evaluation, and saving 
    of the Logistic Regression model for stroke risk prediction.

    Loads preprocessed data, trains the model, prints performance metrics, 
    saves the confusion matrix and trained model.
    """

    # Retrieve training and test data splits
    X_train, X_test, y_train, y_test = preprocess_data() 
    # Train the model
    log_reg_model, y_pred = train_model(X_train, y_train, X_test, y_test)
    # Create and save the confusion matrix
    create_confusion_matrix(y_test, y_pred)
    
    os.makedirs("../../models", exist_ok=True)
    
    joblib.dump(log_reg_model, "../../models/logistic_regression_model.pkl")
    print("Logistic Regression Model saved to /models")

    
    
if __name__ == '__main__':
    train_logistic_regression_model()