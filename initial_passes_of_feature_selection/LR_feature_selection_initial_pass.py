from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score # not using roc auc because this is a regression problem
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import numpy as np
import pandas as pd

# Load dataset
data = pd.read_csv('processed.csv')

# Separate train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    data.drop(labels=['class'], axis=1),
    data['class'],
    test_size=0.3,
    random_state=0)

# Perform backward elimination using SequentialFeatureSelector with Linear Regression
sfs1 = SFS(LinearRegression(), 
           k_features='best', 
           forward=False, 
           floating=False, 
           verbose=2,
           scoring='r2',  # Use 'r2' for Linear Regression
           cv=3)

sfs1 = sfs1.fit(np.array(X_train), y_train)
