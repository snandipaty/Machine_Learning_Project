#pip install numpy
#used for mathematical operations on arrays
import numpy as np

#pip install pandas
#used for working with data sets; has functions for analysing, cleaning, exploring etc.
import pandas as pd

#pip install matplotlib
#used for creating static, animated and interactive visualisations in Python
import matplotlib.pyplot as plt

#pip install seaborn
#data visualisation library, based on matplotlib
import seaborn as sns

#pip install scikit-learn
#provides supervised learning algorithms
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

#read dataset into variable
dataset = pd.read_csv("mushrooms.csv")

print(dataset.describe())

"""
--- Data Preprocessing ---
add some comments what your code does and why
"""

#code here

"""
--- Feature Selection --- 
add some comments what your code does and why
"""

#code here

"""
--- Data Splitting ---
data train set must be at least 70%, preferably 80%
inside Data train set, the validation set is subsetting the training data
testing is usually 5% - 10%, bigger than validation

data in the code below is split into 80% training (10% validation set), 20% test
these variables may change depending which is more effective
as mentioned above, the data training set must be at least 70%

to make the new data sets the same (not shuffled randomly each time command is called)
use `random st={an integer}` optional parameter; 
{an integer} represents the splitting seed - data is split differently for each integer
"""

train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=2)

#split training data into training and validation
train_data, validation_data = train_test_split(train_data, test_size=0.1, random_state=2)

#print(len(train_data), len(test_data), len(validation_data))


"""
--- Model Training --

"""