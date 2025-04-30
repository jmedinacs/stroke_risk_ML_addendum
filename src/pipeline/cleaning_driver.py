'''
Created on Apr 28, 2025

@author: jarpy
'''

from cleaning import data_cleaning as dClean
from utils import data_io

# Load the raw data from the raw data folder
data = dClean.load_raw_data("../../data/raw/stroke_data.csv")

# Inspect the raw data
dClean.inspect_data(data)

# Impute missing BMI values
data = dClean.impute_missing_bmi(data)

# Inspect data to confirm that null BMI values were replaced
dClean.inspect_data(data)

# Inspect categorical values distribution
dClean.inspect_categorical_distribution(data)

# Remove rare 'other' gender
data = dClean.remove_rare_gender(data)

#Inspect data
dClean.inspect_data(data)

# Convert text field data into lower case and trimmed version
dClean.standardize_text_fields(data)

# Confirm that field data are now standardized
dClean.inspect_categorical_distribution(data)

# Remove full-row duplicates (if any)
data = dClean.remove_duplicates(data)

# Save the cleaned processed data
data_io.save_clean_data(data)



if __name__ == '__main__':
    pass