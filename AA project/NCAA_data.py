import os 
import numpy as np 
import pandas

raw_data = pandas.read_csv(os.getcwd() + "/AA project/snoozleQuery.csv")
total_team = set()
total_team_index = {}
home_team = np.unique(raw_data['Home Team'])
vis_team = np.unique(raw_data[' Vis Team'])
count = 0

for t in home_team:
    total_team.add(t)
    total_team_index[t] = count
    count = count + 1

for v in vis_team:
    if v not in total_team:
        total_team.add(v)
        total_team_index[v] = count
        count = count + 1

match_up = raw_data[['Home Team',' Vis Team']]

weight_mat_2 = np.zeros(shape = (len(total_team_index),len(total_team_index)))
for index, row in match_up.iterrows():
        weight_mat_2[total_team_index[row["Home Team"]],total_team_index[row[" Vis Team"]]] = 1
        weight_mat_2[total_team_index[row[" Vis Team"]],total_team_index[row["Home Team"]]] = 1

np.savetxt('NCAA.out', weight_mat_2)