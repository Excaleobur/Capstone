import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the CSV file into a Pandas DataFrame
df = pd.read_csv(r'Capstone\JeevSauce\Phase2\data\merged_G_data_onlycombine_noCollegeStats.csv')

columns_to_exclude = ['Player', 'School', 'College', 'Ht']

# Select the columns you want to normalize (excluding any non-numeric columns)
columns_to_normalize = [col for col in df.columns if col not in columns_to_exclude]

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit the scaler to the selected columns and transform the data
df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])

# Save the normalized data to a new CSV file
df.to_csv(r'Capstone\JeevSauce\Phase2\data\normalized_merged_G_data_onlycombine_noCollegeStats.csv', index=False)

print("Data has been normalized and saved to 'normalized_merged_G_data_onlycombine_noCollegeStats.csv'.")
