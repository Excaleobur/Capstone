import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, f1_score, multilabel_confusion_matrix
import matplotlib.pyplot as plt
import joblib
import seaborn as sns
from sklearn.svm import SVC

df = pd.read_csv('Amir/Data/grouped.csv')

# Select features and the target variable
X = df[['player_game_count', 'attempts', 'avoided_tackles', 'breakaway_attempts', 'breakaway_percent', 'breakaway_yards', 'designed_yards', 'drops', 'elu_recv_mtf', 'elu_rush_mtf', 'elu_yco', 'elusive_rating', 'explosive', 'first_downs', 'fumbles', 'gap_attempts', 'grades_hands_fumble', 'grades_offense', 'grades_offense_penalty', 'grades_pass', 'grades_pass_block', 'grades_pass_route', 'grades_run', 'grades_run_block', 'longest', 'penalties', 'rec_yards', 'receptions', 'routes', 'run_plays', 'scramble_yards', 'scrambles', 'targets', 'total_touches', 'touchdowns', 'yards', 'yards_after_contact', 'yco_attempt', 'ypa', 'yprr', 'zone_attempts']]  # Replace with actual features
#X = df[[ 'attempts','elusive_rating', 'explosive', 'first_downs', 'fumbles', 'grades_run', 'touchdowns', 'yards', 'yards_after_contact',  'ypa']]  # Replace with actual features
#X = df[['attempts', 'yards', 'touchdowns', 'ypa', 'breakaway_yards', 'avoided_tackles', 'elu_rush_mtf', 'yards_after_contact', 'receptions', 'rec_yards', 'targets']]
y = df[['Match', 'PB']]  # Replace with actual target variable

# Convert to NumPy arrays
X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# AdaBoost Model
ab_model = AdaBoostClassifier(n_estimators=500, random_state=42)
multi_target_ab = MultiOutputClassifier(ab_model)

multi_target_ab.fit(X_train, y_train)  # Training
ab_y_pred = multi_target_ab.predict(X_test)

def evaluate_model(y_true, y_pred):
    num_targets = y_true.shape[1]
    accuracies = []
    f1_scores = []
    confusion_matrices = []

    for i in range(num_targets):
        accuracy = accuracy_score(y_true[:, i], y_pred[:, i])
        f1 = f1_score(y_true[:, i], y_pred[:, i])
        cm = confusion_matrix(y_true[:, i], y_pred[:, i])

        accuracies.append(accuracy)
        f1_scores.append(f1)
        confusion_matrices.append(cm)

        print(f"Target {i+1} - Accuracy: {accuracy:.2f}, F1 Score: {f1:.2f}")
        sns.heatmap(cm, annot=True, fmt='d')
        plt.title(f'Confusion Matrix for Target {i+1}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()

    return accuracies, f1_scores, confusion_matrices

# Evaluate the AdaBoost model
accuracies, f1_scores, confusion_matrices = evaluate_model(y_test, ab_y_pred)

#print(f"Accuracy: {accuracy:.2f}")
#print(f"F1 Score: 0.6")
#print("Confusion Matrices:")
#for i, cm in enumerate(confusion_matrices):
    #print(f"Target {i+1}:")
    #print(cm)
