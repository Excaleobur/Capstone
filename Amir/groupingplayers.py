import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your CSV file
df = pd.read_csv('Amir/Data/normalized_RB_data_with_PB1.csv')

# Grouping by 'player' (and 'age' if you have this column)
# Aggregating other statistics based on their mean (or any other suitable aggregation method)
sorteddf = df.sort_values(by='player')


# Saving the grouped DataFrame to a new CSV file
sorteddf.to_csv('Amir/Data/grouped.csv', index=False)