import pandas as pd

# Load the CSV files with raw string paths
df_new = pd.read_csv(r'Capstone\JeevSauce\Phase2\data\normalized_merged_G_data_onlycombine_noCollegeStats.csv')
df_old = pd.read_csv(r'Capstone\JeevSauce\Phase2\includeNoCombine\trends.csv')

# Rename the player name column in df_old to match df_new
df_old.rename(columns={'player': 'Player'}, inplace=True)

# Merge the two dataframes on the 'Player' column (outer join to include all players)
merged_df = pd.merge(df_new, df_old, on='Player', how='outer')

# Fill missing values (NaN) in all columns (except 'Player' or 'name') with 0
columns_to_fill_with_0 = merged_df.columns.difference(['Player', 'name'])
merged_df[columns_to_fill_with_0] = merged_df[columns_to_fill_with_0].fillna(0)

# Save the merged dataframe to a new CSV file
merged_df.to_csv(r'Capstone\JeevSauce\Phase2\includeNoCombine\INCFinal.csv', index=False)

print("Merged data saved to 'Capstone\JeevSauce\Phase2\includeNoCombine\INCFinal.csv'.")
