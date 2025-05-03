'''
Created on May 2, 2025

@author: jarpy
'''

from cleaning.data_preprocess import preprocess_data, normalize_data
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import classification_report, roc_auc_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt 
import joblib 
import os 


def train_model(X_train, y_train, X_test, y_test):
    """
    Trains a K-Nearest Neighbors (KNN) model and evaluates it on the test set.

    Computes predictions, probability scores, classification report, and ROC AUC.

    Args:
        X_train (pd.DataFrame or np.ndarray): Normalized feature matrix for training (SMOTE-applied).
        y_train (pd.Series): Balanced training labels.
        X_test (pd.DataFrame or np.ndarray): Normalized feature matrix for testing.
        y_test (pd.Series): True labels for test set.

    Returns:
        Tuple[KNeighborsClassifier, np.ndarray]: Trained KNN model and predicted test labels.
    """
    # Initialize the model and define parameter
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train, y_train)
    
    y_pred = knn_model.predict(X_test)
    y_prob = knn_model.predict_proba(X_test)[:,1]
    
    print("\nClassification Report (KNN):")
    print(classification_report(y_test, y_pred, digits=3))
    
    knn_auc = roc_auc_score(y_test, y_prob)
    print(f"ROC AUC: {knn_auc:.3f}")
    
    return knn_model, y_pred

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
    plt.title("Confusion Matrix - KNN")
    plt.tight_layout()
    plt.show()
    
    # Save the figure
    os.makedirs("../../outputs/figures", exist_ok=True)
    plt.savefig("../../outputs/figures/confusion_matrix_knn.png", dpi=300)
    print("Confusion matrix saved to /outputs/figures/")

def train_knn_model():
    """
    """
    
    # Retrieve training and test data splits
    X_train, X_test, y_train, y_test = preprocess_data()
    # Normalize the X training and test data
    X_train, X_test, scaler = normalize_data(X_train, X_test, save_path="../../models/knn_scaler.pkl")
    
    knn_model, y_pred = train_model(X_train, y_train, X_test, y_test)
    
    create_confusion_matrix(y_test, y_pred)
    
    # Save the model and scaler
    joblib.dump(knn_model, "../../models/knn_model.pkl")
    joblib.dump(scaler, "../../models/knn_scaler.pkl")
    print("KNN model and scaler saved.")
    
    
if __name__ == '__main__':
    train_knn_model()