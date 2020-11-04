import os
from pathlib import Path
import pandas as pd
from itertools import takewhile

path = "jetris/ALLCombinedVariables_timeseriesPupilEvolution.csv"
destination_directory = "jetris/preprocessed"


data = pd.read_csv("./jetris/ALLCombinedVariables_timeseriesPupilEvolution.csv")

unique_game_ids = data["gameID"].unique()

for id in unique_game_ids:
  print(id)
  frame = data.loc[data['gameID'] == id]
  frame.to_csv(f"jetris/games_raw/{id}.csv")

