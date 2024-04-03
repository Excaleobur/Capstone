import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your CSV file
df = pd.read_csv('Amir/Data/merged_RB_data.csv')

dfattempt = df[df['attempts'] > 0]
dfyards = df[df['yards'] > 0]

Q1 = df['yards'].quantile(0.25)
Q3 = df['yards'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
dfyards = df[(df['yards'] >= lower_bound) & (df['yards'] <= upper_bound)]

dfyards.to_csv('Amir/Data/filtered_RB_data.csv', index=False)


df['attempts'].hist(bins=20)
plt.title('Distribution of Attempts')
plt.xlabel('Attempts')
plt.ylabel('Frequency')
plt.show()

plt.scatter(df['attempts'], df['yards'])
plt.title('Attempts vs Yards')
plt.xlabel('Attempts')
plt.ylabel('Yards')
plt.show()

sns.boxplot(x='explosive', y='yards', data=df)
plt.title('Yards by Explosiveness')
plt.show()

corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Replace 'season' with your actual time-related column
average_yards_per_season = df.groupby('season')['yards'].mean()
average_yards_per_season.plot(kind='line')
plt.title('Average Yards per Season')
plt.xlabel('Season')
plt.ylabel('Average Yards')
plt.show()
