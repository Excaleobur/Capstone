import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the CSV file into a Pandas DataFrame
df = pd.read_csv('Amir/Data/merged_RB_data.csv')

columns_to_exclude = ['player', 'player_id', 'player_game_count', 'attempts','Match']

# Select the columns you want to normalize (excluding any non-numeric columns)
# For example, if you want to normalize all columns except 'Year':
columns_to_normalize = [col for col in df.columns if col not in columns_to_exclude]

# Initialize the StandardScaler
scaler = StandardScaler()
# Fit the scaler to the selected columns and transform the data
df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])

# Save the normalized data to a new CSV file
df.to_csv('Amir/Data/normalized_RB_data.csv', index=False)

print("Data has been normalized and saved to 'normalized_RB_data.csv'.")