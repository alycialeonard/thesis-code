# Get all possible combinations of appliances from MICS data to get data about counts of them.
# Author: Alycia Leonard, Energy and Power Group, University of Oxford
# Contact: alycia.leonard@eng.ox.ac.uk

import pandas as pd
from itertools import combinations
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.ticker import PercentFormatter

datapath = "D:\\hh_ongridandoffgrid.csv"

# Placeholder list for possible combinations,
comb_lists = []
comb_dicts = []
comb_counts = []

# Questions concerning each appliance
appliances = ['HC7B', 'HC9A', 'HC9B', 'HC9C', 'HC9D', 'HC11', 'HC12']

# Appliances default dict (all 2 = all no before changes)
apps = {'HC7B': 2, 'HC9A': 2, 'HC9B': 2, 'HC9C': 2, 'HC9D': 2, 'HC11': 2, 'HC12': 2}

# Get a list of lists (comb_lists) and list of dictionaries (combins) for all combinations of ownership
for L in range(0, len(appliances)+1):
    for subset in combinations(appliances, L):
        # Append to list of combinations
        comb_lists.append(list(subset))
        # Get a copy of the default appliance dict
        a = apps.copy()
        # For each item in the appliance list
        for i in list(subset):
            # Mark ownership in the dictionary
            a[i] = 1
        # Append dictionary to the list of dictionaries (combins)
        comb_dicts.append(a)

# Bring in dataset
df = pd.read_csv(datapath, dtype=str)

# Get relevant subsets of data
df_rural = df.loc[df['HH6'] == '2']
df_urban = df.loc[df['HH6'] == '1']

df_n = df.loc[df['HH7'] == '2']
df_s = df.loc[df['HH7'] == '3']
df_e = df.loc[df['HH7'] == '1']
df_w = df.loc[df['HH7'] == '4']

df_n_rural = df.loc[(df['HH7'] == '2') & (df['HH6'] == '2')]
df_n_urban = df.loc[(df['HH7'] == '2') & (df['HH6'] == '1')]
df_s_rural = df.loc[(df['HH7'] == '3') & (df['HH6'] == '2')]
df_s_urban = df.loc[(df['HH7'] == '3') & (df['HH6'] == '1')]
df_e_rural = df.loc[(df['HH7'] == '1') & (df['HH6'] == '2')]
df_e_urban = df.loc[(df['HH7'] == '1') & (df['HH6'] == '1')]
df_w_rural = df.loc[(df['HH7'] == '4') & (df['HH6'] == '2')]
df_w_urban = df.loc[(df['HH7'] == '4') & (df['HH6'] == '1')]

df_kai = df.loc[df['HH7A'] == '11']
df_ken = df.loc[df['HH7A'] == '12']
df_kon = df.loc[df['HH7A'] == '13']
df_bom = df.loc[df['HH7A'] == '21']
df_kam = df.loc[df['HH7A'] == '22']
df_koi = df.loc[df['HH7A'] == '23']
df_por = df.loc[df['HH7A'] == '24']
df_ton = df.loc[df['HH7A'] == '25']
df_bo_ = df.loc[df['HH7A'] == '31']
df_bon = df.loc[df['HH7A'] == '32']
df_moy = df.loc[df['HH7A'] == '33']
df_puj = df.loc[df['HH7A'] == '34']
df_w_r = df.loc[df['HH7A'] == '41']
df_w_u = df.loc[df['HH7A'] == '42']

df_q1 = df.loc[df['windex5'] == '1']
df_q2 = df.loc[df['windex5'] == '2']
df_q3 = df.loc[df['windex5'] == '3']
df_q4 = df.loc[df['windex5'] == '4']
df_q5 = df.loc[df['windex5'] == '5']

df_north_q4 = df.loc[(df['windex5'] == '4') & (df['HH7'] == '2')]
df_north_q5 = df.loc[(df['windex5'] == '5') & (df['HH7'] == '2')]
df_south_q4 = df.loc[(df['windex5'] == '4') & (df['HH7'] == '3')]
df_south_q5 = df.loc[(df['windex5'] == '5') & (df['HH7'] == '3')]
df_east_q4 = df.loc[(df['windex5'] == '4') & (df['HH7'] == '1')]
df_east_q5 = df.loc[(df['windex5'] == '5') & (df['HH7'] == '1')]
df_west_q4 = df.loc[(df['windex5'] == '4') & (df['HH7'] == '4')]
df_west_q5 = df.loc[(df['windex5'] == '5') & (df['HH7'] == '4')]

# Set what subset we are looking at
df = df_west_q5

# For each combination of appliances
for i in comb_dicts:
    # Get the subset of the df with this combination
    c = df[(df['HC7B'] == str(i["HC7B"])) & (df['HC9A'] == str(i["HC9A"])) & (df['HC9B'] == str(i["HC9B"])) & (df['HC9C'] == str(i["HC9C"])) & (df['HC9D'] == str(i["HC9D"])) & (df['HC11'] == str(i["HC11"])) & (df['HC12'] == str(i["HC12"]))]
    # Get the count of households with that combination (i.e. the length of the df subset)
    count = len(c.index)
    # Add the count to the dictionary for that combination, and to the list of counts
    i['count'] = count
    comb_counts.append(count)

print('Total number of households in sample: ' + str(sum(comb_counts)))

# Plot counts
x = list(range(1, len(comb_counts)+1))
plt.scatter(x, comb_counts)
plt.show(block=False)

# Create dataframe out of comb_dicts
df_results = pd.DataFrame.from_dict(comb_dicts)
# Sort by count
df_results = df_results.sort_values('count', ascending=False)

# Save results as CSV
df_results.to_csv("D:\\combs_q5_west.csv")

# Print the 10 combs we're comparing
# print(df_results.loc[(df_results['HC7B'] == 2) & (df_results['HC9A'] == 2) & (df_results['HC9B'] == 2) & (df_results['HC9C'] == 2) & (df_results['HC9D'] == 2) & (df_results['HC11'] == 2) & (df_results['HC12'] == 2)])
# print(df_results.loc[(df_results['HC7B'] == 2) & (df_results['HC9A'] == 2) & (df_results['HC9B'] == 2) & (df_results['HC9C'] == 2) & (df_results['HC9D'] == 2) & (df_results['HC11'] == 2) & (df_results['HC12'] == 1)])
# print(df_results.loc[(df_results['HC7B'] == 1) & (df_results['HC9A'] == 2) & (df_results['HC9B'] == 2) & (df_results['HC9C'] == 2) & (df_results['HC9D'] == 2) & (df_results['HC11'] == 2) & (df_results['HC12'] == 2)])
# print(df_results.loc[(df_results['HC7B'] == 1) & (df_results['HC9A'] == 2) & (df_results['HC9B'] == 2) & (df_results['HC9C'] == 2) & (df_results['HC9D'] == 2) & (df_results['HC11'] == 2) & (df_results['HC12'] == 1)])
# print(df_results.loc[(df_results['HC7B'] == 2) & (df_results['HC9A'] == 1) & (df_results['HC9B'] == 2) & (df_results['HC9C'] == 2) & (df_results['HC9D'] == 2) & (df_results['HC11'] == 2) & (df_results['HC12'] == 1)])
# print(df_results.loc[(df_results['HC7B'] == 1) & (df_results['HC9A'] == 1) & (df_results['HC9B'] == 2) & (df_results['HC9C'] == 2) & (df_results['HC9D'] == 2) & (df_results['HC11'] == 2) & (df_results['HC12'] == 1)])
# print(df_results.loc[(df_results['HC7B'] == 1) & (df_results['HC9A'] == 1) & (df_results['HC9B'] == 2) & (df_results['HC9C'] == 2) & (df_results['HC9D'] == 1) & (df_results['HC11'] == 2) & (df_results['HC12'] == 1)])
# print(df_results.loc[(df_results['HC7B'] == 1) & (df_results['HC9A'] == 1) & (df_results['HC9B'] == 1) & (df_results['HC9C'] == 2) & (df_results['HC9D'] == 1) & (df_results['HC11'] == 2) & (df_results['HC12'] == 1)])
# print(df_results.loc[(df_results['HC7B'] == 1) & (df_results['HC9A'] == 1) & (df_results['HC9B'] == 1) & (df_results['HC9C'] == 1) & (df_results['HC9D'] == 1) & (df_results['HC11'] == 2) & (df_results['HC12'] == 1)])
# print(df_results.loc[(df_results['HC7B'] == 1) & (df_results['HC9A'] == 1) & (df_results['HC9B'] == 1) & (df_results['HC9C'] == 1) & (df_results['HC9D'] == 1) & (df_results['HC11'] == 1) & (df_results['HC12'] == 1)])

# Transform list of combinations this into a list of dicts where 1 = own, 2 = not owned.
#for c in combs:
    # For each appliance owned in each combination:
#    a = apps.copy()
#    for i in c:
#        # Mark ownership
#        a[i] = 1
#    # Append dictionary to combins
#    combins.append(a)

# Get only combins with non-zero counts
#combins_2 = [i for i in combins if i['count'] > 0]

# Get only combin_counts with non-zero counts
#combin_counts_2 = [i for i in combin_counts if i > 0]

#print(sum(combin_counts_2))

# Print total number of households in all combins
# print(sum(combin_counts_2))

# Get combs with more than 1% of sample using
#print(sum(combin_counts)/100)
#combins_thresh = [i for i in combins if i['count'] > sum(combin_counts_2)/100]

# counts > 10, > 50, > 100, > 200 (for full dataset)
#combins_10 = [i for i in combins if i['count'] > 10]
#combins_50 = [i for i in combins if 50 < i['count'] < 101]
#combins_100 = [i for i in combins if 100 < i['count'] < 201]
#combins_200 = [i for i in combins if i['count'] > 200]

#print(combins_50)
#print(combins_100)
#print(combins_200)