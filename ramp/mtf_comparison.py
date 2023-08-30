# Get a month (d = 31) of MTF days from a sum of n random Narayan days (n = # of households)
# Author: Alycia Leonard, Energy and Power Group, University of Oxford
# Contact: alycia.leonard@eng.ox.ac.uk

from numpy import genfromtxt, add, zeros
import random
import pandas as pd

tier = 5
data_path = "D:\\Narayan_MTF_Demand\\Days\\T" + str(tier) + "_days\\T" + str(tier) + "_"

n = 133
d = 31

# List to hold lists of profiles
days = []

# For each day in the month
for i in range(d):
    # Create placeholder for day's data
    day = zeros(1440)
    # For each connection in the town
    for j in range(n):
        # Get a random day of data at the defined tier
        x = random.randint(0, 364)
        data = genfromtxt(data_path + str(x) + ".csv", delimiter=',')
        # Add it to represent that person
        day = add(day, data)
    # Append day to days
    days.append(day)

# Turn days into a dataframe and export to csv
df = pd.DataFrame(days)
df.to_csv("D:\\T_" + str(tier) + ".csv", header=False)


