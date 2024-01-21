import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

dataset = pd.read_csv("mushrooms.csv")
#print(dataset.head())

"""
--- Data Splitting ---
data train set must be at least 70%, preferably 80%
inside Data train set, the validation set is subsetting the training data
testing is usually 5% - 10%, bigger than validation

split data into 80% training (10% validation set), 20% test
these variables may change depending which is more effective
as mentioned above, the data training set must be at least 70%

to make the new data sets the same (not shuffled randomly each time command is called)
use `random st=10` optional parameter
"""
train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=10)

print(test_data)