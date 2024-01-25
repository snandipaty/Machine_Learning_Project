import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
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
csv_file_path = 'processed.csv'
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
X_train_knn, X_test_knn, y_train_knn, y_test_knn = train_test_split(
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

sfs1 = sfs1.fit(np.array(X_train_knn), y_train_knn)



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

sfs2 = sfs2.fit(np.array(X_train_tree), y_train_tree)

#RANDOM FORREST

X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(
    data.drop(labels=['class'], axis=1),
    data['class'],
    test_size=0.3,
    random_state=0)

# Perform backward elimination using SequentialFeatureSelector with Random Forest
sfs3 = SFS(RandomForestClassifier(), 
           k_features=5, 
           forward=False, 
           floating=False, 
           verbose=2,
           scoring='roc_auc',
           cv=3)

sfs3 = sfs3.fit(np.array(X_train_rf), y_train_rf)




#K NEAREST NEIGHBOURS
selected_X_train = X_train_knn.iloc[:, list(sfs1.k_feature_idx_)]
selected_X_test = X_test_knn.iloc[:, list(sfs1.k_feature_idx_)]

# Initialize and fit the KNN regression model
knn_model = KNeighborsRegressor(n_neighbors=5)  # You can adjust the number of neighbors (n_neighbors) as needed
knn_model.fit(selected_X_train, y_train_knn)

# Make predictions on the test set
y_pred_knn = knn_model.predict(selected_X_test)



# Calculate residuals
residuals_knn = y_test_knn - y_pred_knn

# Create a residual plot for KNN
sns.scatterplot(x=y_pred_knn, y=residuals_knn)
plt.title('Residual Plot (KNN)')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.savefig('graphs/residual_plot_knn.png', dpi=200)

# Actual vs Predicted for KNN
plt.scatter(y_test_knn, y_pred_knn)
plt.title('Actual vs. Predicted Values (KNN)')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.savefig('graphs/actual_vs_predicted_knn.png', dpi=200)

# Learning Curve for KNN
train_sizes_knn, train_scores_knn, test_scores_knn = learning_curve(
    knn_model, selected_X_train, y_train_knn, cv=3, scoring='neg_mean_squared_error')

train_scores_mean_knn = -np.mean(train_scores_knn, axis=1)
test_scores_mean_knn = -np.mean(test_scores_knn, axis=1)

plt.plot(train_sizes_knn, train_scores_mean_knn, label='Training error (KNN)')
plt.plot(train_sizes_knn, test_scores_mean_knn, label='Validation error (KNN)')
plt.title('Learning Curve (KNN)')
plt.xlabel('Training Set Size')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.savefig('graphs/learning_curve_knn', dpi=200)

# Confusion Matrix for KNN
cm_knn = confusion_matrix(y_test_knn, y_pred_knn)
sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix (KNN)')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('graphs/confusion_matrix_knn', dpi=200)

#PERFORMANCE METRICS KNN
knn_accuracy = accuracy_score(y_test_knn, y_pred_knn)
knn_precision = precision_score(y_test_knn, y_pred_knn)
knn_recall = recall_score(y_test_knn, y_pred_knn)
knn_f1 = f1_score(y_test_knn, y_pred_knn)

print("PERFORMANCE METRICS KNN")
print(f'Accuracy: {knn_accuracy:.2f}')
print(f'Precision: {knn_precision:.2f}')
print(f'Recall: {knn_recall:.2f}')
print(f'F1 Score: {knn_f1:.2f}')



#DECISION TREE


selected_X_train_tree = X_train_tree.iloc[:, list(sfs2.k_feature_idx_)]
selected_X_test_tree = X_test_tree.iloc[:, list(sfs2.k_feature_idx_)]


# Initialize and fit the Decision Tree regression model
tree_model = DecisionTreeRegressor(random_state=0)  # You can adjust hyperparameters as needed
tree_model.fit(selected_X_train_tree, y_train_tree)

# Make predictions on the test set
y_pred_tree = tree_model.predict(selected_X_test_tree)

# Calculate residuals
residuals_tree = y_test_tree - y_pred_tree

# Create a residual plot for Decision Tree
sns.scatterplot(x=y_pred_tree, y=residuals_tree)
plt.title('Residual Plot (Decision Tree)')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.savefig('graphs/residual_plot_dt.png', dpi=200)

# Actual vs Predicted for Decision Tree
plt.scatter(y_test_tree, y_pred_tree)
plt.title('Actual vs. Predicted Values (Decision Tree)')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.savefig('graphs/actual_vs_predicted_dt.png', dpi=200)

# Learning Curve for Decision Tree
train_sizes_tree, train_scores_tree, test_scores_tree = learning_curve(
    tree_model, selected_X_train_tree, y_train_tree, cv=3, scoring='neg_mean_squared_error')

train_scores_mean_tree = -np.mean(train_scores_tree, axis=1)
test_scores_mean_tree = -np.mean(test_scores_tree, axis=1)


plt.plot(train_sizes_tree, train_scores_mean_tree, label='Training error (Decision Tree)')
plt.plot(train_sizes_tree, test_scores_mean_tree, label='Validation error (Decision Tree)')
plt.title('Learning Curve (Decision Tree)')
plt.xlabel('Training Set Size')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.savefig('graphs/learning_curve_dt.png', dpi=200)


# Confusion Matrix for DECISION TREE
cm_tree = confusion_matrix(y_test_tree, y_pred_tree)
sns.heatmap(cm_tree, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix (RF)')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('graphs/confusion_matrix_tree', dpi=200)


#PERFORMANCE METRICS
tree_accuracy = accuracy_score(y_test_tree, y_pred_tree)
tree_precision = precision_score(y_test_tree, y_pred_tree)
tree_recall = recall_score(y_test_tree, y_pred_tree)
tree_f1 = f1_score(y_test_tree, y_pred_tree)

print("PERFORMANCE METRICS RANDOM FORREST")
print(f'Accuracy: {tree_accuracy:.2f}')
print(f'Precision: {tree_precision:.2f}')
print(f'Recall: {tree_recall:.2f}')
print(f'F1 Score: {tree_f1:.2f}')




#K RANDOM FORREST
selected_X_train_rf = X_train_rf.iloc[:, list(sfs3.k_feature_idx_)]
selected_X_test_rf = X_test_rf.iloc[:, list(sfs3.k_feature_idx_)]

# Initialize and fit the Decision Tree regression model
rf_model = RandomForestClassifier(random_state=0)
rf_model.fit(selected_X_train_rf, y_train_rf)

# Make predictions on the test set
y_pred_rf = rf_model.predict(selected_X_test_rf)


# Calculate residuals
residuals_rf = y_test_rf - y_pred_rf

sns.scatterplot(x=y_pred_rf, y=residuals_rf)
plt.title('Residual Plot (Random Forest)')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.savefig('graphs/residual_plot_rf.png', dpi=200)


train_sizes_rf, train_scores_rf, test_scores_rf = learning_curve(
    rf_model, selected_X_train_rf, y_train_rf, cv=3, scoring='neg_mean_squared_error')

train_scores_mean_rf = -np.mean(train_scores_rf, axis=1)
test_scores_mean_rf = -np.mean(test_scores_rf, axis=1)

plt.plot(train_sizes_rf, train_scores_mean_rf, label='Training error (RF)')
plt.plot(train_sizes_rf, test_scores_mean_rf, label='Validation error (RF)')
plt.title('Learning Curve (RF)')
plt.xlabel('Training Set Size')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.savefig('graphs/learning_curve_rf', dpi=200)

cm_rf = confusion_matrix(y_test_rf, y_pred_rf)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix (RF)')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('graphs/confusion_matrix_rf', dpi=200)


#PERFORMANCE METRICS
rf_accuracy = accuracy_score(y_test_rf, y_pred_rf)
rf_precision = precision_score(y_test_rf, y_pred_rf)
rf_recall = recall_score(y_test_rf, y_pred_rf)
rf_f1 = f1_score(y_test_rf, y_pred_rf)

print("PERFORMANCE METRICS DECISION TREE")
print(f'Accuracy: {rf_accuracy:.2f}')
print(f'Precision: {rf_precision:.2f}')
print(f'Recall: {rf_recall:.2f}')
print(f'F1 Score: {rf_f1:.2f}')