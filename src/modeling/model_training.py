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
from sklearn.metrics import classification_report, roc_auc_score, precision_score, recall_score
from sklearn.metrics._classification import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import RandomizedSearchCV
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from xgboost import plot_importance 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.preprocessing import StandardScaler 
import matplotlib.pyplot as plt
import shap 
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
plt.close()

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
plt.close()


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
plt.close()

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
plt.close()

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
plt.close()

# ─────────────────────────────────────────────
# Tuning XGBoost
# ─────────────────────────────────────────────

xgb_clf = XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,
    random_state=42
)

param_dist = {
    'n_estimators': [100, 200, 300],          # More trees = more learning (but more overfit risk)
    'learning_rate': [0.01, 0.05, 0.1],       # Lower = slower, more stable learning
    'max_depth': [3, 5, 7],                   # Controls tree complexity (deeper = more interaction capture)
    'subsample': [0.7, 0.8, 1.0],             # Randomly sample rows for each tree (helps reduce overfitting)
    'colsample_bytree': [0.7, 0.8, 1.0],      # Randomly sample features for each tree
    'scale_pos_weight': [1, 5, 10, 20]        # Class imbalance control: ratio of negative/positive
}



# Search across 50 random combinations
random_search = RandomizedSearchCV(
    estimator=xgb_clf,
    param_distributions=param_dist,
    n_iter=50,
    scoring='recall',
    cv=5,
    verbose=2,
    n_jobs=-1,
    random_state=42
)

random_search.fit(X_train_resampled, y_train_resampled)

best_model = random_search.best_estimator_

y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

from sklearn.metrics import classification_report, roc_auc_score

print(classification_report(y_test, y_pred, digits=3))
print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.3f}")

tuned_cm = confusion_matrix(y_test, y_pred)
tuned_disp = ConfusionMatrixDisplay(confusion_matrix=tuned_cm, display_labels=["No Stroke", "Stroke"])
tuned_disp.plot(cmap="Oranges")
plt.title("Confusion Matrix – XGBoost (Tuned for Recall)")
plt.tight_layout()
plt.savefig("../../select_viz/confusion_matrix_xgboost_tuned.png", dpi=300)
print("Tuned confusion matrix saved.")
#plt.show()
plt.close()

joblib.dump(best_model, "../../models/xgboost_tuned.pkl")
print("Tuned XGBoost model saved to /models")

plt.savefig("../../select_viz/confusion_matrix_xgboost_tuned.png", dpi=300)
print("Tuned confusion matrix saved to /select_viz")

thresholds = [0.5, 0.4, 0.3, 0.25, 0.2]

for thresh in thresholds:
    preds_thresh = (y_proba >= thresh).astype(int)
    
    precision = precision_score(y_test, preds_thresh)
    recall = recall_score(y_test, preds_thresh)
    f1 = f1_score(y_test, preds_thresh)
    cm = confusion_matrix(y_test, preds_thresh)

    print(f"\nThreshold: {thresh}")
    print(f"Recall: {recall:.3f}, Precision: {precision:.3f}, F1: {f1:.3f}")
    print(f"Confusion Matrix:\n{cm}")

# Check dtypes
print(X_test.dtypes)

# STEP 1: Convert test data to float64 for SHAP compatibility
X_test_fixed = X_test.astype('float64')

# STEP 2: Compute SHAP values using the tuned model
import shap
explainer = shap.Explainer(best_model, X_test_fixed)
shap_values = explainer(X_test_fixed)

# STEP 3: Load original cleaned dataset to restore metadata like heart_disease and id
original_df = load_clean_data()

# STEP 4: Combine predictions, true labels, and metadata
import pandas as pd

xgb_results = X_test.copy()
xgb_results["prob_stroke"] = best_model.predict_proba(X_test_fixed)[:, 1]
xgb_results["true_stroke"] = y_test.values
xgb_results["heart_disease"] = original_df.loc[X_test.index, "heart_disease"].values
xgb_results["id"] = original_df.loc[X_test.index, "id"].values

# STEP 5: Filter to people with heart disease and sort by predicted stroke probability
heart_cases = xgb_results[xgb_results["heart_disease"] == 1]
heart_cases_sorted = heart_cases.sort_values(by="prob_stroke", ascending=False)

# STEP 6: Show top candidates and pick the highest-risk one
print(heart_cases_sorted[["id", "prob_stroke", "true_stroke"]].head())

# Pick the top person’s row index
row_idx = heart_cases_sorted.index[0]

# Find the positional index of the selected person
position = X_test.index.get_loc(row_idx)

# Then safely pass that to SHAP
shap.plots.waterfall(shap_values[position])

plt.figure()
shap.plots.waterfall(shap_values[position], show=False)
plt.title("SHAP – High Risk Example")
plt.tight_layout()
plt.savefig("../../select_viz/shap_high_risk.png", dpi=300)




# Sort full test set by lowest predicted probability
low_risk_cases = xgb_results.sort_values(by="prob_stroke", ascending=True)

# Preview a few
print(low_risk_cases[["id", "prob_stroke", "true_stroke", "heart_disease", "hypertension"]].head())

low_idx = low_risk_cases.index[0]
position = X_test.index.get_loc(low_idx)

shap.plots.waterfall(shap_values[position])


plt.figure()
shap.plots.waterfall(shap_values[X_test.index.get_loc(low_idx)], show=False)
plt.title("SHAP – Low Risk Example")
plt.tight_layout()
plt.savefig("../../select_viz/shap_low_risk.png", dpi=300)



# Filter moderate-risk patients
moderate_cases = xgb_results[
    (xgb_results["prob_stroke"] >= 0.3) &
    (xgb_results["prob_stroke"] <= 0.5)
].sort_values(by="prob_stroke", ascending=False)

# View a few
print(moderate_cases[["id", "prob_stroke", "true_stroke", "age", "bmi", "hypertension", "heart_disease"]].head())

# Pick one
mod_idx = moderate_cases.index[0]
mod_pos = X_test.index.get_loc(mod_idx)

# Plot SHAP
shap.plots.waterfall(shap_values[mod_pos])

plt.figure()
shap.plots.waterfall(shap_values[mod_pos], show=False)
plt.title("SHAP – Moderate Risk Example")
plt.tight_layout()
plt.savefig("../../select_viz/shap_moderate_risk.png", dpi=300)

from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt

# Plot PDP for 'age'
fig, ax = plt.subplots(figsize=(8, 6))
PartialDependenceDisplay.from_estimator(
    best_model,               # your tuned XGBoost model
    X_test_fixed,             # numeric test set (float64)
    features=["age"],         # name of the column you want to analyze
    kind="average",           # show average predicted probability
    grid_resolution=100,      # number of age values to test across range
    ax=ax
)

plt.title("Partial Dependence Plot – Age vs Stroke Probability")
plt.tight_layout()
plt.savefig("../../select_viz/pdp_age.png", dpi=300)
plt.show()


from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt

# PDP for avg_glucose_level
fig, ax = plt.subplots(figsize=(8, 6))
PartialDependenceDisplay.from_estimator(
    best_model,
    X_test_fixed,
    features=["avg_glucose_level"],
    kind="average",
    grid_resolution=100,
    ax=ax
)

plt.title("Partial Dependence Plot – Glucose Level vs Stroke Probability")
plt.tight_layout()
plt.savefig("../../select_viz/pdp_glucose.png", dpi=300)
plt.show()


from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt

# PDP for bmi
fig, ax = plt.subplots(figsize=(8, 6))
PartialDependenceDisplay.from_estimator(
    best_model,
    X_test_fixed,
    features=["bmi"],
    kind="average",
    grid_resolution=100,
    ax=ax
)

plt.title("Partial Dependence Plot – BMI vs Stroke Probability")
plt.tight_layout()
plt.savefig("../../select_viz/pdp_bmi.png", dpi=300)
plt.show()


