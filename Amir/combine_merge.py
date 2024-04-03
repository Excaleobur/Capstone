import pandas as pd 
from sklearn.preprocessing import MinMaxScaler

df_main = pd.read_csv('/Users/amirrezarafati/Downloads/CapsotneModel/RB/Repo/Capstone/Amir/Data/filtered_RB_data.csv')  # Update the path
df_combine_stats = pd.read_csv('Amir/Data/RB_Combine_Stats.csv')  # Update the path

merged_df = pd.merge(df_main, df_combine_stats, left_on='player', right_on='Name', how='left')


merged_data = merged_df.drop(columns=['Name','Year','College', 'POS'])

merged_data.to_csv('Amir/Data/combine_added.csv', index=False) 

column_to_move = 'Match'

# Create a new column order, excluding 'column_to_move' first, then adding it at the end
new_column_order = [col for col in merged_data.columns if col != column_to_move] + [column_to_move]

# Reorder the DataFrame columns
merged_data = merged_data[new_column_order]

merged_data.to_csv('Amir/Data/combine_added1.csv', index=False) 

columns_to_exclude = ['player', 'player_id','Match']

# Select the columns you want to normalize (excluding any non-numeric columns)
# For example, if you want to normalize all columns except 'Year':
columns_to_normalize = [col for col in merged_data.columns if col not in columns_to_exclude]

# Initialize the StandardScaler
scaler = MinMaxScaler(feature_range=(0, 1))

# Fit the scaler to the selected columns and transform the data
merged_data[columns_to_normalize] = scaler.fit_transform(merged_data[columns_to_normalize])

merged_data.fillna(0, inplace=True)

column_to_move = 'Match'

# Create a new column order, excluding 'column_to_move' first, then adding it at the end
new_column_order = [col for col in merged_data.columns if col != column_to_move] + [column_to_move]

# Reorder the DataFrame columns
merged_data = merged_data[new_column_order]

merged_data.to_csv('Amir/Data/combine_added.csv', index=False) 

print(merged_data.columns)