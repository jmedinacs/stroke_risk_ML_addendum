"""
data_cleaning.py

This module contains functions for inspecting, cleaning, and preparing the raw Stroke Risk dataset
for machine learning modeling. Cleaning steps include handling missing values, removing rare categories,
and encoding categorical variables for supervised learning.

Author: John Medina
Date: 2025-04-28
Project: Stroke Risk ML Addendum

Notes:
- Missing 'bmi' values will be imputed using the median, due to skewness in the BMI distribution.
- Rows with gender 'Other' will be removed due to extremely low occurrence.
- Categorical variables will be standardized for clean encoding.
"""

import pandas as pd
import os


def inspect_data(df):
    """
    Display dataset info, summary statistics, and missing values.

    Parameters:
    - df (DataFrame): Input dataset.

    Returns:
    - None
    """
    print(df.info())
    print(df.describe())
    print(df.isnull().sum())


def clean_data(df):
    """
    Run all data cleaning steps in sequence.

    Parameters:
    - df (DataFrame): Raw stroke dataset.

    Returns:
    - DataFrame: Cleaned dataset ready for EDA and modeling.
    """
    
    print("\nStarting cleaning process...")
    
    print("\nStep 1: Removing rare gender value 'Other' (single occurrence in dataset)")    
    df = remove_rare_gender(df)
    
    print("\nStep 2: Imputing missing BMI values with median value.")
    df = impute_missing_bmi(df)
    
    print("\nStep 3: Standardizing text fields (lowercase, trimmed)")
    df = standardize_text_fields(df)
    
    print("\nStep 4: Removing duplicate rows")
    df = remove_duplicates(df)
    
    print("\n\nData cleaning complete.")
    return df


def impute_missing_bmi(df):
    """
    Fill missing 'bmi' values using the median of the column.

    Parameters:
    - df (DataFrame): Input dataset.

    Returns:
    - DataFrame: Dataset with missing BMI values imputed.
    """
    median_bmi = df['bmi'].median()
    df['bmi'] = df['bmi'].fillna(median_bmi)
    print(f"Missing BMI values filled with median: {median_bmi:.2f}")
    return df


def inspect_categorical_distribution(df):
    """
    Print value counts and percentage breakdown for all categorical columns.

    Parameters:
    - df (DataFrame): Input dataset.

    Returns:
    - None
    """
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        print(f"Value counts for '{col}':")
        print(df[col].value_counts())
        print("\nPercentage breakdown:")
        print((df[col].value_counts(normalize=True) * 100).round(2))
        print("-" * 50)


def remove_rare_gender(df):
    """
    Remove rows where 'gender' is 'other', due to extreme rarity.

    Parameters:
    - df (DataFrame): Input dataset.

    Returns:
    - DataFrame: Dataset with rare gender category removed.
    """
    before_rows = df.shape[0]
    df = df[df['gender'] != 'Other'].copy()
    after_rows = df.shape[0]
    print(f"Removed {before_rows - after_rows} rows with rare 'Other' gender value.")
    return df


def standardize_text_fields(df):
    """
    Convert all object columns to lowercase and strip whitespace.

    Parameters:
    - df (DataFrame): Input dataset.

    Returns:
    - DataFrame: Dataset with cleaned text columns.
    """
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df[col] = df[col].str.strip().str.lower()
    print(f"Standardized {len(cat_cols)} text columns: lowercased and trimmed.")
    return df


def remove_duplicates(df):
    """
    Drop duplicate rows from the dataset.

    Parameters:
    - df (DataFrame): Input dataset.

    Returns:
    - DataFrame: Dataset with duplicates removed.
    """
    before_rows = df.shape[0]
    df = df.drop_duplicates()
    after_rows = df.shape[0]
    print(f"Removed {before_rows - after_rows} duplicate rows.")
    return df
    
    
    
    