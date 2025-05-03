"""
data_preprocess.py

Preprocessing pipeline for stroke risk classification.

This module loads the cleaned dataset and performs the following steps:
- Drops statistically insignificant features (based on EDA/chi-square tests)
- Splits features and target
- One-hot encodes categorical variables
- Performs stratified train/test split
- Applies SMOTE oversampling to the training set only

Returns preprocessed and ready-to-train datasets for machine learning models.
"""


from utils.data_io import load_clean_data
import pandas as pd 
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE 
from sklearn.preprocessing import StandardScaler 
import joblib 


def drop_insignificant_features(df):
    """
    Drops features found to be statistically insignificant in stroke prediction.

    Args:
        df (pd.DataFrame): Cleaned input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with dropped columns (id, gender, Residence_type).
    """
    df = df.drop(columns=["id","gender","Residence_type"])
    return df

def split_features_target(df):
    """
    Splits the DataFrame into features (X) and target (y).

    Args:
        df (pd.DataFrame): Preprocessed DataFrame with all features.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Feature matrix X and target vector y.
    """
    X = df.drop(columns=["stroke"])
    y = df["stroke"]
    return X, y

def encode_categoricals(X):
    """
    Performs one-hot encoding on categorical features.

    Args:
        X (pd.DataFrame): Feature matrix before encoding.

    Returns:
        pd.DataFrame: One-hot encoded feature matrix.
    """
    X = pd.get_dummies(X, drop_first=True)
    return X

def train_test_stratified_split(X, y):
    """
    Performs stratified train/test split to preserve class distribution.

    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target vector.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: 
            X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    return X_train, X_test, y_train, y_test

def apply_smote(X_train, y_train):
    """
    Applies SMOTE oversampling to balance the training dataset.

    Args:
        X_train (pd.DataFrame): Training feature matrix.
        y_train (pd.Series): Training target vector.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Resampled X_train and y_train with balanced classes.
    """
    # Initialize SMOTE
    smote = SMOTE(random_state=42)
    
    # Apply SMOTE to the TRAINING SET ONLY
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    return X_train_resampled, y_train_resampled    
    

def preprocess_data():
    """
    Full preprocessing pipeline for stroke risk classification.

    Steps:
    - Loads cleaned dataset using `load_clean_data()`
    - Drops features found statistically insignificant (id, gender, Residence_type)
    - Splits data into features (X) and target (y)
    - One-hot encodes categorical features
    - Performs stratified train/test split (80/20)
    - Applies SMOTE oversampling to training data only to address class imbalance

    Returns:
        X_train_resampled (pd.DataFrame): Feature matrix for training (balanced)
        X_test (pd.DataFrame): Feature matrix for testing (original distribution)
        y_train_resampled (pd.Series): Balanced target vector for training
        y_test (pd.Series): Target vector for testing (original distribution)
    """
    # Load cleaned version of the dataset
    df = load_clean_data()
    # Drop the features that were identified as insignificant from EDA
    df = drop_insignificant_features(df)
    # Separate features from the target
    X, y = split_features_target(df)
    # One-hot encode categorical data
    X_encoded = encode_categoricals(X)
    # Split the set into training and test sets (80/20 split)
    X_train, X_test, y_train, y_test = train_test_stratified_split(X_encoded, y)
    # Apply SMOTE on training data
    X_train_resampled, y_train_resampled = apply_smote(X_train, y_train)
    
  
    print("\n✅ Shapes")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"X_train_resampled shape: {X_train_resampled.shape}")

    print("\n📊 Target distribution before SMOTE (y_train):")
    print(y_train.value_counts(normalize=True).round(3))

    print("\n📊 Target distribution after SMOTE (y_train_resampled):")
    print(y_train_resampled.value_counts(normalize=True).round(3))
    
    return X_train_resampled, X_test, y_train_resampled, y_test

def normalize_data(X_train, X_test, save_path=None):
    """
    Normalizes feature data using StandardScaler.

    Args:
        X_train (pd.DataFrame): Training feature matrix.
        X_test (pd.DataFrame): Test feature matrix.
        save_path (str, optional): If provided, saves the fitted scaler to disk.

    Returns:
        Tuple[np.ndarray, np.ndarray, StandardScaler]: Normalized X_train and X_test, and the scaler.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    if save_path:
        joblib.dump(scaler, save_path)
        print(f"Scaler saved to {save_path}")
        
    return X_train_scaled, X_test_scaled, scaler
    

if __name__ == '__main__':
    preprocess_data()