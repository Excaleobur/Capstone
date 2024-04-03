import pandas as pd

# List of CSV files to merge
csv_files = [
        'Capstone/JeevSauce/G/GData/1996_G_B.csv',
    'Capstone/JeevSauce/G/GData/1998_G_B.csv',
    'Capstone/JeevSauce/G/GData/2001_G_B.csv',
    'Capstone/JeevSauce/G/GData/2002_G_B.csv',
    'Capstone/JeevSauce/G/GData/2003_G_B.csv',
    'Capstone/JeevSauce/G/GData/2004_G_B.csv',
    'Capstone/JeevSauce/G/GData/2006_G_B.csv',
    'Capstone/JeevSauce/G/GData/2007_G_B.csv',
    'Capstone/JeevSauce/G/GData/2008_G_B.csv',
    'Capstone/JeevSauce/G/GData/2009_G_B.csv',
    'Capstone/JeevSauce/G/GData/2010_G_B.csv',
    'Capstone/JeevSauce/G/GData/2011_G_B.csv',
    'Capstone/JeevSauce/G/GData/2012_G_B.csv',
    'Capstone/JeevSauce/G/GData/2013_G_B.csv',
    'Capstone/JeevSauce/G/GData/2014_G_B.csv',
    'Capstone/JeevSauce/G/GData/2015_G_B.csv',
    'Capstone/JeevSauce/G/GData/2016_G_B.csv',
    'Capstone/JeevSauce/G/GData/2017_G_B.csv',
    'Capstone/JeevSauce/G/GData/2018_G_B.csv',
    'Capstone/JeevSauce/G/GData/2019_G_B.csv',
    'Capstone/JeevSauce/G/GData/2020_G_B.csv',
    'Capstone/JeevSauce/G/GData/2021_G_B.csv',
    'Capstone/JeevSauce/G/GData/2022_G_B.csv',
    'Capstone/JeevSauce/G/GData/2023_G_B.csv'
]

# Initialize an empty DataFrame to store the merged data
merged_data = pd.DataFrame()

# Loop through each CSV file and concatenate them into the merged_data DataFrame
for file in csv_files:
    df = pd.read_csv(file)
    
    # Filter rows where the 'position' column is 'HB'
    df = df[df['position'] == 'G']
    
    # Extract the year from the file name and add it as a new column 'YOP'
    year = file.split('/')[-1].split('_')[0]  # Extracts the year from the filename
    df['YOP'] = year  # Add the year as a new column

    # Concatenate the filtered DataFrame with merged_data
    merged_data = pd.concat([merged_data, df], ignore_index=True)

# Drop the 'team name' and 'position' columns
merged_data = merged_data.drop(columns=['team_name', 'position'])

# Save the merged data to a new CSV file
merged_data.to_csv('Capstone/JeevSauce/Phase2/includeNoCombine/merged_G_data_with_years.csv', index=False)

print("Merged CSV files, filtered for 'G' position, added 'YOP', and removed 'team name' and 'position' columns. Added years")
