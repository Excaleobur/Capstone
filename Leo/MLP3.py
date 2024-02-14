import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score
from imblearn.over_sampling import SMOTE

# Read the dataset
ncaa = pd.read_csv('finalnormalizedncaaWithCombine.csv')
ncaa = ncaa.fillna(0)

# Convert the specified categorical columns to string type
for col in ['position', 'team_name']:
    ncaa[col] = ncaa[col].astype(str)

# Identify categorical columns for one-hot encoding
categorical_columns = ['position', 'team_name']

# Drop the target column 'nfl' before applying transformations
features = ncaa.drop(['nfl', 'player'], axis=1)

# Apply one-hot encoding to categorical columns
column_transformer = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(), categorical_columns)
    ],
    remainder='passthrough'
)

# Transform the dataset with the column_transformer
features_encoded = column_transformer.fit_transform(features)
features_encoded = features_encoded.toarray()

# Separate the target variable
y = ncaa['nfl']

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(features_encoded, y, test_size=0.2, random_state=42)

# Apply SMOTE to the training set
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# RandomForestClassifier with initial setup
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_smote, y_train_smote)

# dictionary defines the grid of hyperparameters to be tested
param_grid = {
    # The number of trees in the forest. Two values are tested: 100 and 200
    'n_estimators': [100, 200],
    # The maximum depth of the trees. It tests three scenarios: no limit (None), and trees with a maximum depth of 10 and 20
    'max_depth': [None, 10, 20],
    # smallest number of samples (or data points) that must be present in a node of the tree before it can be split into two new nodes
    'min_samples_split': [2, 5],
    # This is the minimum number of samples a leaf (the end point of a tree where a prediction is made) must have
    'min_samples_leaf': [1, 2]
}

# GridSearchCV is a function that performs hyperparameter tuning in order to find the optimal hyperparameters for a model.
# rf is the estimator, param_grid is the dictionary of hyperparameters, cv is the number of folds in the cross-validation process,
#  scoring metric is how it should evaluate the performance of the model, and n_jobs is the number of cores to use in parallel
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='f1', n_jobs=-1)

# Fit the grid search to the data
grid_search.fit(X_train_smote, y_train_smote)

# provides the model that performed best during the hyperparameter search process
best_rf = grid_search.best_estimator_

# Evaluate the best model from grid search
y_pred = best_rf.predict(X_test)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("Best Random Forest Parameters: ", grid_search.best_params_)
print("F1 Score: ", f1)
print("Confusion Matrix: \n", cm)

# Cross-validation for robustness check
cv_scores = cross_val_score(best_rf, X_train_smote, y_train_smote, cv=5, scoring='f1')
print("Average F1 Score from CV: ", np.mean(cv_scores))
