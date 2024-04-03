import pandas as pd

# Load the CSV files with raw string paths
df_new = pd.read_csv(r'Capstone\JeevSauce\Phase2\data\normalized_merged_G_data_onlycombine_noCollegeStats.csv')
df_old = pd.read_csv(r'Capstone\JeevSauce\G\GData\normalized_G_data.csv')

# Rename the player name column in df_old to match df_new
df_old.rename(columns={'player': 'Player'}, inplace=True)

# Merge the two dataframes on the 'Player' column using an outer join
merged_df = pd.merge(df_new, df_old, on='Player', how='outer')

# Replace NaN values with zeros
merged_df.fillna(0, inplace=True)

# Save the merged dataframe to a new CSV file
merged_df.to_csv(r'Capstone\JeevSauce\Phase2\includeNoCombine\collegeAndCombineNormalized.csv', index=False)

print("Merged data saved to 'Capstone\JeevSauce\Phase2\includeNoCombine\collegeAndCombineNormalized.csv'.")
