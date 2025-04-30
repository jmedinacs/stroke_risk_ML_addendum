'''
Created on Apr 29, 2025

@author: jarpy
'''

import pandas as pd 
import os 

def load_clean_data(filepath="../../data/processed/clean_stroke_data.csv"):
    """
    Loads the cleaned stroke dataset from disk.

    Parameters:
    - filepath (str): Path to the cleaned CSV file

    Returns:
    - DataFrame: Cleaned stroke dataset
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found at: {filepath}")
    
    df = pd.read_csv(filepath)
    return df

def save_clean_data(df, filepath="../../data/processed/clean_stroke_data.csv"):
    """
    Saves the cleaned dataset to a CSV file.

    Parameters:
    - df (DataFrame): Cleaned stroke dataset
    - filepath (str): Output file path for saving the CSV
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"✅ Cleaned data saved to: {filepath}")