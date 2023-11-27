import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_fscore_support

# Load the dataset
df = pd.read_csv('Capstone/JeevSauce/G/GData/normalized_G_data.csv')

# Select features and target variable
X = df[['player_game_count', 'block_percent', 'declined_penalties', 'hits_allowed', 'hurries_allowed', 'non_spike_pass_block', 'penalties', 'pressures_allowed', 'sacks_allowed', 'snap_counts_block', 'snap_counts_ce', 'snap_counts_lt', 'snap_counts_pass_block', 'snap_counts_pass_play', 'snap_counts_run_block']]
y = df['Match']

# Convert to NumPy arrays
X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate the Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Training
clf.fit(X_train, y_train)

# Prediction
y_pred = clf.predict(X_test)

# Metrics
cm = confusion_matrix(y_test, y_pred)
precision, recall, fscore, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')

print("Random Forest Confusion Matrix:\n", cm)
print(f"Random Forest Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {fscore:.4f}")
