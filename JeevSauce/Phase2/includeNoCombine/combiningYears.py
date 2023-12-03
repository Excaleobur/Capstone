import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

# Load the dataset
df = pd.read_csv('Capstone/JeevSauce/Phase2/includeNoCombine/normalized_G_data.csv')

# Identify the columns to calculate the trend
trend_columns = ['player_game_count', 'block_percent', 'declined_penalties', 'grades_offense', 'grades_pass_block', 'grades_run_block', 'hits_allowed', 'hurries_allowed', 'non_spike_pass_block', 'penalties', 'pressures_allowed', 'sacks_allowed', 'snap_counts_block', 'snap_counts_pass_block', 'snap_counts_pass_play', 'snap_counts_run_block']

# Function to calculate the overall trend
def calculate_overall_trend(group):
    # Initialize trend list
    trends = []

    # Calculate trend for each column if more than one year of data
    if len(group['YOP'].unique()) > 1:
        for column in trend_columns:
            X = group['YOP'].values.reshape(-1, 1)  # Reshape for sklearn
            y = group[column].fillna(0).values  # Fill NaN with 0
            reg = LinearRegression().fit(X, y)
            trends.append(reg.coef_[0])  # Append the slope (trend)
    else:
        trends = [0] * len(trend_columns)  # Zero trends for a single year

    # Calculate overall trend as average of individual trends
    overall_trend = np.mean(trends)

    # Combine average stats with overall trend
    average_stats = group.mean()
    average_stats['Overall_Trend'] = overall_trend

    return average_stats

# Group by player, calculate overall trend and average stats
final_df = df.fillna(0).groupby('player').apply(calculate_overall_trend).reset_index()

# Reordering columns to place 'Overall_Trend' before 'YOP'
columns = final_df.columns.tolist()
columns.insert(columns.index('YOP'), columns.pop(columns.index('Overall_Trend')))
final_df = final_df[columns]

# Save the final dataframe to a new CSV file
final_df.to_csv('Capstone/JeevSauce/Phase2/includeNoCombine/trends.csv', index=False)

print("Processed data saved to 'trends.csv'.")
