import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_fscore_support
from imblearn.over_sampling import RandomOverSampler

# Load the dataset
df = pd.read_csv('Capstone/JeevSauce/Phase2/data/collegeAndCombineNormalized.csv')

# Separate numeric and non-numeric columns
numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
non_numeric_columns = df.select_dtypes(exclude=[np.number]).columns.tolist()

# Apply one-hot encoding to non-numeric columns
df_one_hot_encoded = pd.get_dummies(df[non_numeric_columns])

# Combine the numeric data with the one-hot encoded data
df_combined = pd.concat([df[numeric_columns], df_one_hot_encoded], axis=1)

# Handling NaN values in numeric columns
df_combined[numeric_columns] = df_combined[numeric_columns].fillna(df_combined[numeric_columns].mean())

# Select features and target variable
X = df_combined.drop('Match', axis=1)  # Assuming 'Match' is your target variable
y = df_combined['Match']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Oversample the minority class
oversampler = RandomOverSampler(sampling_strategy='minority', random_state=42)
X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train, y_train)

# Define and train the AdaBoost model
base_classifier = DecisionTreeClassifier(max_depth=1)  # Adjust the depth as needed
ada_classifier = AdaBoostClassifier(base_classifier, n_estimators=50, random_state=42)
ada_classifier.fit(X_train_resampled, y_train_resampled)

# Make predictions on the test set
y_pred = ada_classifier.predict(X_test)

# Calculate F1 score, confusion matrix, and other metrics
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
precision, recall, fscore, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')

print("AdaBoost Model Confusion Matrix:\n", cm)
print(f"AdaBoost Model Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
