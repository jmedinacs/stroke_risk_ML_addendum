"""
eda_driver.py

Main execution script for exploratory data analysis (EDA) on the cleaned stroke risk dataset.

This script loads the cleaned dataset, generates visualizations for numeric and categorical
features, runs chi-squared and point-biserial correlation tests, and saves the results.

Author: John Medina
Date: 2025-04-29
Project: Stroke Risk ML Addendum
"""

from eda.eda_distribution import explore_numerics
from eda.eda_categorical import explore_categoricals
from eda.eda_bivariate_and_chi_square import run_chi_square_test
from eda.eda_continuous_vs_stroke import run_point_biserial
from eda.eda_target_distribution import explore_target_feature
from utils.data_io import load_clean_data


def run_eda():
    # Load cleaned dataset
    data = load_clean_data()

    # ➤ Explore numeric feature distributions
    explore_numerics(data, output_dir=None, show_plot=False)

    # ➤ Explore categorical feature distributions
    explore_categoricals(data, output_dir=None, show_plot=False)

    # ➤ Run chi-squared tests on categorical features vs. stroke
    chi_results = run_chi_square_test(data)
    print("\nChi-Square Test Results:")
    print(chi_results)
    chi_results.to_csv("../../results/chi_square_summary.csv", index=False)

    # ➤ Run point-biserial correlations on continuous features vs. stroke
    biserial_results = run_point_biserial(data)
    print("\nPoint-Biserial Correlation Results:")
    print(biserial_results)
    biserial_results.to_csv("../../results/point_biserial_summary.csv", index=False)

    # ➤ Visualize the distribution of the target variable (stroke)
    explore_target_feature(data, output_dir=None, show_plot=False)


if __name__ == '__main__':
    run_eda()