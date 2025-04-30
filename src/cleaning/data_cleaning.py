"""
data_cleaning.py

This module contains functions for inspecting, cleaning, and preparing the raw Stroke Risk dataset
for machine learning modeling. Cleaning steps include handling missing values, removing rare categories,
and encoding categorical variables for supervised learning.

Author: John Medina
Date: 2025-04-28
Project: Stroke Risk ML Addendum

Notes:
- Missing 'bmi' values will be imputed using the median, due to skewness in BMI distribution.
- Rows with gender 'Other' will be removed due to extremely low occurrence.
- Categorical variables will be encoded appropriately for modeling compatibility.
- Raw data should be located at '../data/raw/stroke_data.csv' relative to script execution.
"""

import pandas as pd
from pandas.tests.generic.test_label_or_level_utils import df
import os 


def load_raw_data(filepath):
    """Load raw stroke risk data."""
    data = pd.read_csv(filepath)
    print(f"Data loaded: {data.shape[0]} rows, {data.shape[1]} columns")
    return data

def inspect_data(df):
    """Print basic information about the dataset."""
    print(df.info())
    print(df.describe())
    print(df.isnull().sum())
    
def clean_data(df):
    """Apply all cleaning steps to the dataset."""
    #1.
    
def impute_missing_bmi(df):
    """
    Fill missing BMI values using the median of the BMI column (28.1).
    
    Parameters:
    df (DataFrame): Input dataframe containing the stroke risk dataset.
    
    Returns:
    Dataframe: Updated dataframe with missing BMI values imputed.
    """
    
    median_bmi = df['bmi'].median()
    df['bmi'] = df['bmi'].fillna(median_bmi)
    print(f"Missing BMI values filled with median: {median_bmi:.2f}")
    return df

def inspect_categorical_distribution(df):
    """
    Inspect value distributions of all categorical (object-type) columns.

    Parameters:
    df (DataFrame): Input dataframe containing the stroke risk dataset.

    Returns:
    None
    """
    
    cat_cols = df.select_dtypes(include=['object']).columns
    
    for col in cat_cols:
        print(f"Value counts for '{col}':")
        print(df[col].value_counts())
        print("\n Percentage Breakdown:")
        print(df[col].value_counts(normalize=True).round(4)*100)
        print("-" * 50)
        
def remove_rare_gender(df):
    """
    Remove rows where 'gender' is 'other', due to extreme rarity (exactly 1 occurrence)
    
    Parameters:
    df (DataFrame): Input dataframe containing the stroke risk dataset.
    
    Returns:
    DataFrame: Updated dataframe with rare gender category removed.
    """
    
    before_rows = df.shape[0]
    df = df[df['gender']!= 'Other']
    after_rows = df.shape[0]
    print(f"Removed {before_rows - after_rows} rows with rate 'Other' gender value.")
    return df
    
def standardize_text_fields(df):
    """
    Standardize all object (text) columns to lowercase and trimmed (remove leading/trailing spaces).

    Parameters:
    df (DataFrame): Input dataframe.

    Returns:
    DataFrame: Updated dataframe with standardized text fields.
    """    
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df[col] = df[col].str.strip().str.lower()
    print(f"Standardized text fields: lower case and trimmed for {len(cat_cols)} columns.")
    return df
    
def remove_duplicates(df):
    """
    Identify and remove full-row duplicate records.

    Parameters:
    df (DataFrame): Input dataframe.

    Returns:
    DataFrame: Updated dataframe with duplicates removed.
    """
    before_rows = df.shape[0]
    df = df.drop_duplicates()
    after_rows = df.shape[0]
    print(f"Removed {before_rows - after_rows} duplicate rows.")
    return df    

    
    
    
    