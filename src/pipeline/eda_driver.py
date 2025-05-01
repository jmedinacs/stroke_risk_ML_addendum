'''
Created on Apr 29, 2025

@author: jarpy
'''

from eda.eda_distribution import explore_numerics
from utils import data_io
from eda.eda_categorical import explore_categoricals
from pickle import FALSE
from eda.eda_bivariate_and_chi_square import run_chi_square_test
from eda.eda_continuous_vs_stroke import run_point_biseral
from eda.eda_target_distribution import explore_target_feature

# Load the clean data
data = data_io.load_clean_data()

# Explore the numeric features of the dataset adn save the visualization
explore_numerics(data,None,False)

# Explore the categorical features of the dataset and save the visualization
explore_categoricals(data, None, False)

# Run the chi square test
chi_results = run_chi_square_test(data)
# Print the results of the Chi-Square test
print("Chi-Square test results:")
print(chi_results)
chi_results.to_csv("../../results/chi_square_summary.csv", index=False)

biserial_results = run_point_biseral(data)
print("Point-Biserial Correlation Results:")
print(biserial_results)
biserial_results.to_csv("../../results/point_biserial_summary.csv", index=False)

explore_target_feature(data, None, True)







if __name__ == '__main__':
    pass