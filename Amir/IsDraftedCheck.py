import pandas as pd

# Load the first CSV file
df1 = pd.read_csv('Amir/Data/DraftedRB_14-23.csv')

# Load the second CSV file
df2 = pd.read_csv('Amir/Data/merged_RB_data.csv')

# Save the modified DataFrame to a new CSV file
df1.to_csv('Amir/Data/DraftedRB_14-23.csv', index=False)

# Extract the names to compare from both DataFrames
names_to_compare = df1['Names'].tolist()

# Add a new column 'Match' to the second DataFrame and initialize it with 0
df2['Match'] = 0

# Iterate through the names in the second DataFrame and update 'Match' to 1 if it matches any name from the first DataFrame
for index, row in df2.iterrows():
    if row['player'] in names_to_compare:
        df2.at[index, 'Match'] = 1

# Save the updated second DataFrame to a new CSV file
df2.to_csv('Amir/data/merged_RB_data.csv', index=False)

print("Comparison and update complete. The updated data is saved.")
