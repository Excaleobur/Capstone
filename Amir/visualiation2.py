import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Step 1: Load the Data
df = pd.read_csv('Amir/Data/combine_added1.csv')

# Step 2: Data Inspection
print(df.head())  # Display the first few rows
print(df.info())  # Get a concise summary of the DataFrame
print(df.isnull().sum())  # Check for missing values

# Step 3: Data Summarization
print(df.describe())  # Summary statistics for numerical columns

# For a specific column, e.g., 40-yard dash times
print("40-yard dash times summary:")
print(df['40Yard'].describe())

# Step 4: Visualization
# Histogram of 40-yard dash times
plt.figure(figsize=(10, 6))
sns.histplot(df['40Yard'], kde=True)
plt.title('Distribution of 40-yard Dash Times')
plt.xlabel('Time (seconds)')
plt.ylabel('Frequency')
plt.show()

# Exclude non-numerical columns except for an ID column if it's numerical and relevant
numerical_df = df.select_dtypes(include=[np.number])

# If you want to exclude the 'ID' column because it's just an identifier, you can drop it
# Assuming your ID column is named 'PlayerID', adjust the name as necessary
if 'Player' in numerical_df.columns:
    numerical_df = numerical_df.drop(['Player'], axis=1)

# Now, use the numerical_df for the correlation matrix heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(numerical_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Numerical Statistics')
plt.show()

plt.figure(figsize=(10, 6))
sns.violinplot(x='match', y='40Yard', data=df)
plt.title('40-yard Dash Times by Match Status')
plt.xlabel('Match (0 = No Match, 1 = Match)')
plt.ylabel('40-yard Dash Time (seconds)')
plt.show()