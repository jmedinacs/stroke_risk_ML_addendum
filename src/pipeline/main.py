"""
main.py

Main execution script for the Stroke Risk ML Addendum project.

This script loads raw data, applies data cleaning steps, and saves the cleaned dataset.
Subsequent EDA and modeling steps are coordinated from this file or relevant drivers.

Author: John Medina
Date: 2025-04-30
Project: Stroke Risk ML Addendum
"""

from cleaning.data_cleaning import clean_data, inspect_data
from utils.data_io import save_clean_data, load_raw_data

# Load raw data
df_raw = load_raw_data("../../data/raw/stroke_data.csv")

inspect_data(df_raw)

# Clean data using full cleaning pipeline
df_clean = clean_data(df_raw)

df_clean.to_csv("../../data/processed/stroke_cleaned.csv", index=False)
print("Cleaned data saved to data/processed")


if __name__ == '__main__':
    pass