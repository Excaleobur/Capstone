import pandas as pd

# Load your CSV files
df1 = pd.read_csv('Amir/Data/normalized_RB_data.csv')  # Replace with your first CSV file path
df2 = pd.read_csv('Amir/Data/ProBowlers14-19.csv') # Replace with your second CSV file path

# Clean up player names in the second dataframe (remove '%' if present)
df2['player_name'] = df2['Player'].str.replace('%', '')

# Create a set of player names from the second dataframe for efficient lookup
player_names_set = set(df2['Player'])

# Check if player's name in the first dataframe matches any name in the second dataframe
# Set 'PB' to 1 if match found, else 0
df1['PB'] = df1['Player'].apply(lambda name: 1 if name in player_names_set else 0)

# Save the modified first dataframe back to the same CSV file (or another one if you prefer)
df1.to_csv('Amir/Data/normalized_RB_data.csv', index=False)
