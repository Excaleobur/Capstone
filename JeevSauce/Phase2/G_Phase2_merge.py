import pandas as pd

# List of CSV files to merge
csv_files = [
    'Capstone/JeevSauce/Phase2/data/2000_combine.csv',

    'Capstone/JeevSauce/Phase2/data/2001_combine.csv',
    'Capstone/JeevSauce/Phase2/data/2002_combine.csv',
    'Capstone/JeevSauce/Phase2/data/2003_combine.csv',
    'Capstone/JeevSauce/Phase2/data/2004_combine.csv',
    'Capstone/JeevSauce/Phase2/data/2005_combine.csv',
    'Capstone/JeevSauce/Phase2/data/2006_combine.csv',
    'Capstone/JeevSauce/Phase2/data/2007_combine.csv',
    'Capstone/JeevSauce/Phase2/data/2008_combine.csv',
    'Capstone/JeevSauce/Phase2/data/2009_combine.csv',
    'Capstone/JeevSauce/Phase2/data/2010_combine.csv',
    'Capstone/JeevSauce/Phase2/data/2011_combine.csv',
    'Capstone/JeevSauce/Phase2/data/2012_combine.csv',
    'Capstone/JeevSauce/Phase2/data/2013_combine.csv',
    'Capstone/JeevSauce/Phase2/data/2014_combine.csv',
    'Capstone/JeevSauce/Phase2/data/2015_combine.csv',
    'Capstone/JeevSauce/Phase2/data/2016_combine.csv',
    'Capstone/JeevSauce/Phase2/data/2017_combine.csv',
    'Capstone/JeevSauce/Phase2/data/2018_combine.csv',
    'Capstone/JeevSauce/Phase2/data/2019_combine.csv',
    'Capstone/JeevSauce/Phase2/data/2020_combine.csv',
    'Capstone/JeevSauce/Phase2/data/2021_combine.csv',
    'Capstone/JeevSauce/Phase2/data/2022_combine.csv',
    'Capstone/JeevSauce/Phase2/data/2023_combine.csv'

    # 'Capstone\JeevSauce\Phase2\data\2001_combine.csv',
    # 'Capstone\JeevSauce\Phase2\data\2002_combine.csv',
    # 'Capstone\JeevSauce\Phase2\data\2003_combine.csv',
    # 'Capstone\JeevSauce\Phase2\data\2004_combine.csv',
    # 'Capstone\JeevSauce\Phase2\data\2005_combine.csv',
    # 'Capstone\JeevSauce\Phase2\data\2006_combine.csv',
    # 'Capstone\JeevSauce\Phase2\data\2007_combine.csv',
    # 'Capstone\JeevSauce\Phase2\data\2008_combine.csv',
    # 'Capstone\JeevSauce\Phase2\data\2009_combine.csv',
    # 'Capstone\JeevSauce\Phase2\data\2010_combine.csv',
    # 'Capstone\JeevSauce\Phase2\data\2011_combine.csv',
    # 'Capstone\JeevSauce\Phase2\data\2012_combine.csv',
    # 'Capstone\JeevSauce\Phase2\data\2013_combine.csv',
    # 'Capstone\JeevSauce\Phase2\data\2014_combine.csv',
    # 'Capstone\JeevSauce\Phase2\data\2015_combine.csv',
    # 'Capstone\JeevSauce\Phase2\data\2016_combine.csv',
    # 'Capstone\JeevSauce\Phase2\data\2017_combine.csv',
    # 'Capstone\JeevSauce\Phase2\data\2018_combine.csv',
    # 'Capstone\JeevSauce\Phase2\data\2019_combine.csv',
    # 'Capstone\JeevSauce\Phase2\data\2020_combine.csv',
    # 'Capstone\JeevSauce\Phase2\data\2021_combine.csv',
    # 'Capstone\JeevSauce\Phase2\data\2022_combine.csv',
    # 'Capstone\JeevSauce\Phase2\data\2023_combine.csv'
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
merged_data = merged_data.drop(columns=['Pos', 'Drafted (tm/rnd/yr)','Player-additional'])

# Save the merged data to a new CSV file
merged_data.to_csv('Capstone\JeevSauce\Phase2\data\merged_G_data_onlycombine_noCollegeStats.csv', index=False)

print("Merged CSV files, filtered for 'G' position, and removed 'team name' and 'position' columns.")
