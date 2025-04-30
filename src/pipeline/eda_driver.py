'''
Created on Apr 29, 2025

@author: jarpy
'''

from eda.eda_distribution import explore_numerics
from utils import data_io
from eda.eda_categorical import explore_categoricals
from pickle import FALSE

# Load the clean data
data = data_io.load_clean_data()

# Explore the numeric features of the dataset
explore_numerics(data,None,False)

explore_categoricals(data, None, True)




if __name__ == '__main__':
    pass