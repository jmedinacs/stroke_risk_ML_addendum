'''
Created on Apr 30, 2025

@author: jarpy
'''

import pandas as pd 
from scipy.stats import chi2_contingency 

def run_chi_square_test(df):
    # Define the features to test
    features_to_test = [
        'gender',
        'hypertension',
        'ever_married',
        'work_type',
        'Residence_type',
        'smoking_status'
    ]
    
    # Initialize a list to store results
    chi_square_results = []
    
    # Loop through each feature
    for feature in features_to_test:
        contingency = pd.crosstab(df[feature], df['stroke'])
        
        try:
            chi2, p, dof, expected = chi2_contingency(contingency)
            chi_square_results.append({
                'feature': feature,
                'chi2_statistic' : chi2,
                'p_value' : p,
                'degrees_of_freedom' : dof 
            })
        except Exception as e:
            print(f"Error with feature '{feature}' : {e}")
          
    result_df = pd.DataFrame(chi_square_results).sort_values('p_value')
    result_df['significant'] = result_df['p_value'] < 0.05
    return result_df 
