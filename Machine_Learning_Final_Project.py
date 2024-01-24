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
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import learning_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score # not using roc auc for regression problems
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import matplotlib.pyplot as plt


import warnings
warnings.simplefilter("ignore", category=FutureWarning)

#read dataset into variable
dataset = pd.read_csv("processed.csv")

#print(dataset.describe())

"""
--- Data Preprocessing ---
add some comments what your code does and why


"""

# Load the CSV file into a pandas DataFrame
csv_file_path = 'mushrooms.csv'
df = pd.read_csv(csv_file_path)

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Iterate over each column in the DataFrame
for column in df.columns:
    # Check if the column has object (string) dtype
    if df[column].dtype == 'object':
        # Use LabelEncoder to transform the string values into numerical values
        df[column] = label_encoder.fit_transform(df[column])

# Save the modified DataFrame back to a CSV file
output_csv_path = 'processed.csv'
df.to_csv(output_csv_path, index=False)

"""
--- Feature Selection --- 
add some comments what your code does and why
"""

data = pd.read_csv('processed.csv')

# Separate train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    data.drop(labels=['class'], axis=1),
    data['class'],
    test_size=0.3,
    random_state=0)

# Perform backward elimination using SequentialFeatureSelector with KNN
sfs1 = SFS(KNeighborsClassifier(), 
           k_features=5, 
           forward=False, 
           floating=False, 
           verbose=2,
           scoring='roc_auc',
           cv=3)

sfs1 = sfs1.fit(np.array(X_train), y_train)

# repeat with decision tree

X_train_tree, X_test_tree, y_train_tree, y_test_tree = train_test_split(
    data.drop(labels=['class'], axis=1),
    data['class'],
    test_size=0.3,
    random_state=0)

# Perform backward elimination using SequentialFeatureSelector with Decision Trees
sfs2 = SFS(DecisionTreeClassifier(), 
           k_features=5, 
           forward=False, 
           floating=False, 
           verbose=2,
           scoring='roc_auc',
           cv=3)

sfs2 = sfs2.fit(np.array(X_train), y_train)

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



"""
--- Model Training --

"""

#K NEAREST NEIGHBOURS
selected_X_train = X_train.iloc[:, list(sfs1.k_feature_idx_)]
selected_X_test = X_test.iloc[:, list(sfs1.k_feature_idx_)]

print("Training data shape:", selected_X_train.shape, y_train.shape)
print("Testing data shape:", selected_X_test.shape, y_test.shape)

# Initialize and fit the KNN regression model
knn_model = KNeighborsRegressor(n_neighbors=5)  # You can adjust the number of neighbors (n_neighbors) as needed
knn_model.fit(selected_X_train, y_train)

# Make predictions on the test set
y_pred_knn = knn_model.predict(selected_X_test)

# Evaluate model performance
mse_knn = mean_squared_error(y_test, y_pred_knn)
r2_knn = r2_score(y_test, y_pred_knn)

print(f'Mean Squared Error (KNN): {mse_knn}')
print(f'R-squared (KNN): {r2_knn}')

# Calculate residuals
residuals_knn = y_test - y_pred_knn

# Create a residual plot for KNN
sns.scatterplot(x=y_pred_knn, y=residuals_knn)
plt.title('Residual Plot (KNN)')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.savefig('graphs/residual_plot_knn.png', dpi=200)

# Actual vs Predicted for KNN
plt.scatter(y_test, y_pred_knn)
plt.title('Actual vs. Predicted Values (KNN)')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.savefig('graphs/actual_vs_predicted_knn.png', dpi=200)

# Learning Curve for KNN
train_sizes_knn, train_scores_knn, test_scores_knn = learning_curve(
    knn_model, selected_X_train, y_train, cv=3, scoring='neg_mean_squared_error')

train_scores_mean_knn = -np.mean(train_scores_knn, axis=1)
test_scores_mean_knn = -np.mean(test_scores_knn, axis=1)

plt.plot(train_sizes_knn, train_scores_mean_knn, label='Training error (KNN)')
plt.plot(train_sizes_knn, test_scores_mean_knn, label='Validation error (KNN)')
plt.title('Learning Curve (KNN)')
plt.xlabel('Training Set Size')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.savefig('graphs/learning_curve_knn', dpi=200)

unique_predicted_values = np.unique(y_pred_knn)
unique_residuals = np.unique(residuals_knn)

print("Unique Predicted Values:", unique_predicted_values)
print("Unique Residuals:", unique_residuals)

#DECISION TREE
from sklearn.tree import DecisionTreeRegressor

# Select the features based on the features selected using SequentialFeatureSelector
selected_X_train_tree = X_train_tree.iloc[:, list(sfs2.k_feature_idx_)]
selected_X_test_tree = X_test_tree.iloc[:, list(sfs2.k_feature_idx_)]

print("Training data shape:", selected_X_train_tree.shape, y_train_tree.shape)
print("Testing data shape:", selected_X_test_tree.shape, y_test_tree.shape)

# Initialize and fit the Decision Tree regression model
dt_model = DecisionTreeRegressor(random_state=0)  # You can adjust hyperparameters as needed
dt_model.fit(selected_X_train_tree, y_train_tree)

# Make predictions on the test set
y_pred_dt = dt_model.predict(selected_X_test_tree)

# Evaluate model performance
mse_dt = mean_squared_error(y_test_tree, y_pred_dt)
r2_dt = r2_score(y_test_tree, y_pred_dt)

print(f'Mean Squared Error (Decision Tree): {mse_dt}')
print(f'R-squared (Decision Tree): {r2_dt}')

# Calculate residuals
residuals_dt = y_test_tree - y_pred_dt

# Create a residual plot for Decision Tree
sns.scatterplot(x=y_pred_dt, y=residuals_dt)
plt.title('Residual Plot (Decision Tree)')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.savefig('graphs/residual_plot_dt.png', dpi=200)

# Actual vs Predicted for Decision Tree
plt.scatter(y_test_tree, y_pred_dt)
plt.title('Actual vs. Predicted Values (Decision Tree)')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.savefig('graphs/actual_vs_predicted_dt.png', dpi=200)

# Learning Curve for Decision Tree
train_sizes_dt, train_scores_dt, test_scores_dt = learning_curve(
    dt_model, selected_X_train_tree, y_train_tree, cv=3, scoring='neg_mean_squared_error')

train_scores_mean_dt = -np.mean(train_scores_dt, axis=1)
test_scores_mean_dt = -np.mean(test_scores_dt, axis=1)

plt.plot(train_sizes_dt, train_scores_mean_dt, label='Training error (Decision Tree)')
plt.plot(train_sizes_dt, test_scores_mean_dt, label='Validation error (Decision Tree)')
plt.title('Learning Curve (Decision Tree)')
plt.xlabel('Training Set Size')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.savefig('graphs/learning_curve_dt.png', dpi=200)

# Unique values in predicted values and residuals for Decision Tree
unique_predicted_values_dt = np.unique(y_pred_dt)
unique_residuals_dt = np.unique(residuals_dt)

print("Unique Predicted Values (Decision Tree):", unique_predicted_values_dt)
print("Unique Residuals (Decision Tree):", unique_residuals_dt)


"""
---> Random Forest
"""

from sklearn.ensemble import RandomForestClassifier

#K NEAREST NEIGHBOURS
selected_X_train_rf = X_train.iloc[:, list(sfs1.k_feature_idx_)]
selected_X_test_rf = X_test.iloc[:, list(sfs1.k_feature_idx_)]

# Initialize and fit the Decision Tree regression model

rf_model = RandomForestClassifier(random_state=0)

rf_model.fit(selected_X_train_rf, y_train_tree)

# Make predictions on the test set
y_pred_rf = rf_model.predict(selected_X_test_rf)

# Evaluate model performance
mse_rf = mean_squared_error(y_test_tree, y_pred_rf)
r2_rf = r2_score(y_test_tree, y_pred_dt)

print(f'Mean Squared Error (Random Forest): {mse_rf}')
print(f'R-squared (Random Forest): {r2_rf}')
# Calculate residuals
residuals_rf = y_test_tree - y_pred_rf

# Unique values in predicted values and residuals for Decision Tree
unique_predicted_values_rf = np.unique(y_pred_rf)
unique_residuals_rf = np.unique(residuals_rf)

print("Unique Predicted Values (Random Forest):", unique_predicted_values_rf)
print("Unique Residuals (Random Forest):", unique_residuals_rf)

# Confusion Matrix for KNN
cm_knn = confusion_matrix(y_test, y_pred_binary)
sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix (KNN)')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Confusion Matrix for Decision Tree
cm_dt = confusion_matrix(y_test_tree, (y_pred_dt > 0.5).astype(int))
sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix (Decision Tree)')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

print("duplicate train set", selected_X_train.duplicated())
print("duplicate test set", selected_X_test.duplicated())