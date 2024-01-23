import csv
from sportsipy.ncaaf.roster import Player
from sportsipy.ncaaf.teams import Teams

# Define the years for which we want to collect data
years = range(2014, 2023)

# Initialize a list to store player data
player_data = []

# Iterate over each year
for year in years:
    # Fetch teams for the year
    teams = Teams(year)
    # Iterate over each team
    for team in teams:
        # Get roster for the team
        roster = team.roster
        # Iterate over each player in the roster
        for player in roster.players:
            # Check if the player's position is Running Back (RB)
            if player.position == 'RB':
                # Fetch detailed player data
                detailed_player = Player(player.player_id)
                # Extract the required information
                player_info = {
                    'name': player.name,
                    'year': year,
                    'height': detailed_player.height,
                    'weight': detailed_player.weight,
                }
                # Append the data to the list
                player_data.append(player_info)

# Writing data to a CSV file
csv_file = "ncaaf_rb_physical_data.csv"
csv_columns = ['name', 'year', 'height', 'weight']

try:
    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        for data in player_data:
            writer.writerow(data)
except IOError:
    print("I/O error while writing the CSV file")
except Exception as e:
    print(f"An error occurred: {e}")

# The data is now written to 'ncaaf_rb_physical_data.csv'
