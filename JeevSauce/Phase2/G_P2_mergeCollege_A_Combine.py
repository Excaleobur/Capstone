import pandas as pd

# List of CSV files to merge
csv_files = [
    'Capstone\JeevSauce\Phase2\data\merged_G_data_onlycombine_noCollegeStats.csv',
    'Capstone\JeevSauce\Phase2\data\merged_G_data_onlycombine_noCollegeStats.csv',

    

]

# Initialize an empty DataFrame to store the merged data
merged_data = pd.DataFrame()

# Loop through each CSV file and concatenate them into the merged_data DataFrame
for file in csv_files:
    df = pd.read_csv(file)
    
    # Filter rows where the 'position' column is 'OG'
    df = df[df['Pos'] == 'OG']
    
    # Concatenate the filtered DataFrame with merged_data
    merged_data = pd.concat([merged_data, df], ignore_index=True)

# Drop the 'team name' and 'position' columns
merged_data = merged_data.drop(columns=['Pos'])

# Save the merged data to a new CSV file
merged_data.to_csv('Capstone\JeevSauce\Phase2\data\merged_G_data_onlycombine_noCollegeStats.csv', index=False)

print("Merged CSV files, filtered for 'G' position, and removed 'team name' and 'position' columns.")
