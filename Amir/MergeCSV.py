import pandas as pd

# List of CSV files to merge
csv_files = [
    'Amir/Data/rushing_summary2014.csv',
    'Amir/Data/rushing_summary2015.csv',
    'Amir/Data/rushing_summary2016.csv',
    'Amir/Data/rushing_summary2017.csv',
    'Amir/Data/rushing_summary2018.csv',
    'Amir/Data/rushing_summary2019.csv',
    'Amir/Data/rushing_summary2020.csv',
    'Amir/Data/rushing_summary2021.csv',
    'Amir/Data/rushing_summary2022.csv'
]

# Initialize an empty DataFrame to store the merged data
merged_data = pd.DataFrame()

# Loop through each CSV file and concatenate them into the merged_data DataFrame
for file in csv_files:
    df = pd.read_csv(file)
    
    # Filter rows where the 'position' column is 'HB'
    df = df[df['position'] == 'HB']
    
    # Concatenate the filtered DataFrame with merged_data
    merged_data = pd.concat([merged_data, df], ignore_index=True)

# Drop the 'team name' and 'position' columns
merged_data = merged_data.drop(columns=['team_name', 'position'])

# Save the merged data to a new CSV file
merged_data.to_csv('Amir/Data/merged_RB_data.csv', index=False)

print("Merged CSV files, filtered for 'HB' position, and removed 'team name' and 'position' columns.")
