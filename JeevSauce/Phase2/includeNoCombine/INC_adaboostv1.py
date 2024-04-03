import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_fscore_support, precision_recall_curve
from imblearn.over_sampling import RandomOverSampler

# Load the dataset
df = pd.read_csv('INCFinal.csv')

# Select features and target variable
features = ['player_game_count', 'block_percent', 'declined_penalties', 'hits_allowed', 'hurries_allowed', 'non_spike_pass_block',
            '40yd', 'Vertical', 'Bench', 'Broad Jump', '3Cone', 'Shuttle',
            'penalties', 'pressures_allowed', 'sacks_allowed', 'snap_counts_block', 'snap_counts_ce', 'snap_counts_lt', 'snap_counts_pass_block', 'snap_counts_pass_play', 'snap_counts_run_block']
X = df[features]
y = df['Match']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Oversample the minority class
oversampler = RandomOverSampler(sampling_strategy='minority', random_state=42)
X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train, y_train)

# Define and train the AdaBoost model
base_classifier = DecisionTreeClassifier(max_depth=1)
ada_classifier = AdaBoostClassifier(base_classifier, n_estimators=50, random_state=42)
ada_classifier.fit(X_train_resampled, y_train_resampled)

# Make predictions on the test set
y_pred = ada_classifier.predict(X_test)

# Calculate F1 score, confusion matrix, and other metrics
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
precision, recall, fscore, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')

# Feature Importance Visualization
feature_importances = ada_classifier.feature_importances_
sns.barplot(x=feature_importances, y=features)
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

# Confusion Matrix Heatmap
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()

# Precision-Recall Curve
y_scores = ada_classifier.decision_function(X_test)
precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
plt.plot(recall, precision, marker='.')
plt.title('Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.tight_layout()
plt.show()

# F1 Score by Number of Estimators
f1_scores = []
estimator_range = range(1, 101, 10)
for n_estimators in estimator_range:
    ada_classifier_temp = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=n_estimators, random_state=42)
    ada_classifier_temp.fit(X_train_resampled, y_train_resampled)
    y_pred_temp = ada_classifier_temp.predict(X_test)
    f1_scores.append(f1_score(y_test, y_pred_temp))

plt.plot(estimator_range, f1_scores, marker='o')
plt.title('F1 Score by Number of Estimators')
plt.xlabel('Number of Estimators')
plt.ylabel('F1 Score')
plt.tight_layout()
plt.show()
