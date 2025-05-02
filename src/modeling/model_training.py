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
from sklearn.metrics._classification import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier 
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from xgboost import plot_importance 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.preprocessing import StandardScaler 
import matplotlib.pyplot as plt
import joblib
import os
from utils.data_io import load_clean_data 


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


# ─────────────────────────────────────────────
# Logistic Regression (Baseline Model)
# ─────────────────────────────────────────────



# Initialize and train logistic regression model
log_reg_model = LogisticRegression(max_iter=1000, random_state=42)
log_reg_model.fit(X_train_resampled, y_train_resampled)

# Predict on the original test set (imbalanced)
y_pred = log_reg_model.predict(X_test)
y_prob = log_reg_model.predict_proba(X_test)[:,1] # probability of stroke (class 1)

# Evaluation metrics
print("\n Classification Report (Logistic Regression): ")
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
print("Confusion matrix saved to /outputs/figures/")
#plt.show()

os.makedirs("../../models", exist_ok=True)

joblib.dump(log_reg_model, "../../models/logistic_regression_model.pkl")
print("Logistic Regression Model saved to /models")


# ─────────────────────────────────────────────
# Random Forest Model
# ─────────────────────────────────────────────


rf_model = RandomForestClassifier(
    n_estimators=100, # Number of trees
    random_state = 42,
    class_weight='balanced'
)
rf_model.fit(X_train_resampled, y_train_resampled)

rf_preds = rf_model.predict(X_test)
rf_probs = rf_model.predict_proba(X_test)[:, 1]  # probability of class 1 (stroke)

print("\nClassification Report (Random Forest):")
print(classification_report(y_test, rf_preds, digits=3))

rf_auc = roc_auc_score(y_test, rf_probs)
print(f"ROC AUC: {rf_auc:.3f}")

joblib.dump(rf_model, "../../models/random_forest_model.pkl")
print("Random Forest model saved.")

# Create the confusion matrix
rf_cm = confusion_matrix(y_test, rf_preds)

# Display and save
rf_disp = ConfusionMatrixDisplay(confusion_matrix=rf_cm, display_labels=["No Stroke", "Stroke"])
rf_disp.plot(cmap="Blues")
plt.title("Confusion Matrix – Random Forest")
plt.tight_layout()

# Save the visualization into the figures folder
plt.savefig("../../outputs/figures/confusion_matrix_random_forest.png", dpi=300)
print("Confusion matrix saved to /outputs/figures")
#plt.show()


# ─────────────────────────────────────────────
# XGBoost Model
# ─────────────────────────────────────────────

xgb_model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    eval_metric='logloss',
    random_state=42
)

xgb_model.fit(X_train_resampled, y_train_resampled)

xgb_preds = xgb_model.predict(X_test)
xgb_probs = xgb_model.predict_proba(X_test)[:, 1]



print("\nClassification Report (XGBoost):")
print(classification_report(y_test, xgb_preds, digits=3))

xgb_auc = roc_auc_score(y_test, xgb_probs)
print(f"ROC AUC: {xgb_auc:.3f}")

joblib.dump(xgb_model, "../../models/xgboost_model.pkl")
print("XGBoost model saved.")

xgb_cm = confusion_matrix(y_test, xgb_preds)
xgb_disp = ConfusionMatrixDisplay(confusion_matrix=xgb_cm, display_labels=["No Stroke", "Stroke"])
xgb_disp.plot(cmap="Blues")
plt.title("Confusion Matrix – XGBoost")
plt.tight_layout()
plt.savefig("../../select_viz/confusion_matrix_xgboost.png", dpi=300)
print("Confusion matrix saved.")
#plt.show()

# === Feature Importance (Gain-Based) ===
importance_scores = xgb_model.get_booster().get_score(importance_type='gain')

# Convert to DataFrame for easier viewing
importance_df = pd.DataFrame({
    'Feature': list(importance_scores.keys()),
    'Importance (Gain)': list(importance_scores.values())
}).sort_values(by='Importance (Gain)', ascending=False)

# Preview top features
print("\nTop Features by Information Gain (XGBoost):")
print(importance_df.head(10))

# Plot feature importance
plot_importance(xgb_model, importance_type='gain', max_num_features=15)
plt.title("Top 15 Feature Importances by Gain (XGBoost)")
plt.tight_layout()
plt.savefig("../../select_viz/feature_importance_xgboost.png", dpi=300)
print("Feature importance chart saved.")
#plt.show()

# ─────────────────────────────────────────────
# K Nearest Neighbor (KNN)
# ─────────────────────────────────────────────

# Normalize numeric features only
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train_resampled)

knn_preds = knn_model.predict(X_test_scaled)
knn_probs = knn_model.predict_proba(X_test_scaled)[:, 1]

print("\nClassification Report (KNN):")
print(classification_report(y_test, knn_preds, digits=3))

knn_auc = roc_auc_score(y_test, knn_probs)
print(f"ROC AUC: {knn_auc:.3f}")

# Save the model adn scaler
joblib.dump(knn_model, "../../models/knn_model.pkl")
joblib.dump(scaler, "../../models/knn_scaler.pkl")
print("KNN model and scaler saved.")

knn_cm = confusion_matrix(y_test, knn_preds)
knn_disp = ConfusionMatrixDisplay(confusion_matrix=knn_cm, display_labels=["No Stroke", "Stroke"])
knn_disp.plot(cmap="Blues")
plt.title("Confusion Matrix – KNN")
plt.tight_layout()
plt.savefig("../../select_viz/confusion_matrix_knn.png", dpi=300)
print("Confusion matrix saved.")
#plt.show()

# ─────────────────────────────────────────────
# Logistic Regression Scaled
# ─────────────────────────────────────────────

logreg_scaled = LogisticRegression(max_iter=1000, random_state=42)
logreg_scaled.fit(X_train_scaled, y_train_resampled)

y_pred_scaled = logreg_scaled.predict(X_test_scaled)
y_prob_scaled = logreg_scaled.predict_proba(X_test_scaled)[:, 1]

print("\nClassification Report (LogReg with Normalization):")
print(classification_report(y_test, y_pred_scaled, digits=3))

logreg_scaled_auc = roc_auc_score(y_test, y_prob_scaled)
print(f"ROC AUC: {logreg_scaled_auc:.3f}")

cm_scaled = confusion_matrix(y_test, y_pred_scaled)
disp_scaled = ConfusionMatrixDisplay(confusion_matrix=cm_scaled, display_labels=["No Stroke", "Stroke"])
disp_scaled.plot(cmap="Blues")
plt.title("Confusion Matrix – LogReg (Normalized)")
plt.tight_layout()
plt.savefig("../../select_viz/confusion_matrix_logreg_normalized.png", dpi=300)
plt.show()


