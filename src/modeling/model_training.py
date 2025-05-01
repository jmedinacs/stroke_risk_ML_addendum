"""
model_training.py

This script trains a baseline Logistic Regression model to predict stroke occurrence
using the cleaned and preprocessed dataset. It includes train/test split, SMOTE
for class imbalance, model training, and performance evaluation.

Author: John Medina
Date: 2025-04-30
Project: Stroke Risk ML Addendum
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import joblib
import os
from utils.data_io import load_clean_data 
from sklearn.metrics._classification import classification_report

# Load cleaned data
df = load_clean_data()

# Drop features not statistically significant
df = df.drop(columns=["id","gender","Residence_type"])

# Separate features (x) and target (y)
X = df.drop(columns=["stroke"])
y = df["stroke"]

# One-hot encode categorical variables
X = pd.get_dummies(X, drop_first=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Check dataset shapes
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")

# Check target distribution before resampling
print("\ny_train class distribution:")
print(y_train.value_counts(normalize=True).round(3))

print("\ny_test class distribution:")
print(y_test.value_counts(normalize=True).round(3))

# Initialize SMOTE - balancing the training data due to low number of stroke occurences
smote = SMOTE(random_state=42)

#Apply SMOTE to the TRAINING SET ONLY
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# 🧾 Confirm the resampled class distribution
print("Resampling complete")
print("Resampled training set distribution:")
print(y_train_resampled.value_counts(normalize=True).round(3))

# Initialize and train logistic regression model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_resampled, y_train_resampled)

# Predict on the original test set (imbalanced)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1] # probability of stroke (class 1)

# Evaluation metrics
print("\n Classification Report (Test Set): ")
print(classification_report(y_test, y_pred, digits=3))

auc = roc_auc_score(y_test, y_prob)
print(f"ROC AUC: {auc: 3f}")

# Create the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Visualize it
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Stroke", "Stroke"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix - Logistic Regression")
plt.tight_layout()

# Save the figure
os.makedirs("../../outputs/figures", exist_ok=True)
plt.savefig("../../outputs/figures/confusion_matrix_logreg.png", dpi=300)
print("✅ Confusion matrix saved to /outputs/figures/")
plt.show()