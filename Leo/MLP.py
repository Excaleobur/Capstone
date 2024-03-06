import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import class_weight
import numpy as np
import tensorflow as tf
from joblib import dump

# Read the dataset
ncaa = pd.read_csv('Leo/finalnormalizedncaaWithCombine.csv')
ncaa = ncaa.fillna(0)

# Convert the specified categorical columns to string type
for col in ['position', 'team_name']:
    ncaa[col] = ncaa[col].astype(str)

# Identify categorical columns for one-hot encoding and handle unknown categories
categorical_columns = ['position', 'team_name']
column_transformer = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
    ],
    remainder='passthrough'
)

# Transform the dataset with the column_transformer
features_encoded = column_transformer.fit_transform(ncaa.drop(['nfl', 'player'], axis=1))
features_encoded = features_encoded.toarray()

# Separate the target variable
y = ncaa['nfl']

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(features_encoded, y, test_size=0.2, random_state=42)

# Define the model architecture
input_dimension = X_train.shape[1]
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(input_dimension,)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Compute class weights and fit the model
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = {i: weights for i, weights in enumerate(class_weights)}
history = model.fit(X_train, y_train, epochs=100, batch_size=50, validation_split=0.2, class_weight=class_weights)

# Save the model and preprocessing object
model.save('leofirstmodel.h5')
dump(column_transformer, 'my_column_transformer.joblib')

# Evaluate the model and print metrics
score = model.evaluate(X_test, y_test, verbose=0)
print('Test accuracy:', score[1])

# Plot the accuracy and loss curves
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# confusion matrix and f1 score
from sklearn.metrics import confusion_matrix, f1_score

# When you want to actually predict stuff with no labels
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)  # Convert to class labels
cm = confusion_matrix(y_test, y_pred_classes)
print(cm)
f1 = f1_score(y_test, y_pred_classes)
print(f1)






