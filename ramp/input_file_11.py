'''
This input file constructs profiles which account for community wealth distribution.
Q1, Q2, and Q3 usage are fabricated. Q4 and Q5 are based on empirical data for usage from those quintiles from region.
Q1: All use task lighting. Combs "None" and "M" (50% each).
Q2: All use task and general lighting. Combs "None", "M", "R", "MR" (25% each).
Q3: All use task and general lighting. Combs "None" + all possible of M, R, T & F (i.e. limit to these appliances).
    Weights based on regional q4 combinations. Distribute remaining weight evenly across combs.
Q4: All use task and general lighting. Combs based on regional quintile 4 appliance combinations from MICS.
Q5: All use task and general lighting. Combs based on regional quintile 5 appliance combinations from MICS.

Results in folder V4
# Author: Alycia Leonard, Energy and Power Group, University of Oxford
# Contact: alycia.leonard@eng.ox.ac.uk
'''

import csv
from core import *
import collections, functools, operator
import random

# INPUT DATA #

# Define sunrise and sunset times in each case
sunrise_dict = {'north': 426, 'south': 424, 'east': 422, 'west': 432}
sunset_dict = {'north': 1125, 'south': 1129, 'east': 1126, 'west': 1133}

# Define the proportions of each region in each quintile
quintiles = {1: {'north': 0.292, 'south': 0.404, 'east': 0.271, 'west': 0.003, 'even': 0.2},
             2: {'north': 0.284, 'south': 0.243, 'east': 0.257, 'west': 0.010, 'even': 0.2},
             3: {'north': 0.244, 'south': 0.198, 'east': 0.253, 'west': 0.049, 'even': 0.2},
             4: {'north': 0.111, 'south': 0.091, 'east': 0.147, 'west': 0.359, 'even': 0.2},
             5: {'north': 0.069, 'south': 0.064, 'east': 0.072, 'west': 0.579, 'even': 0.2}}

# PARAMETERS #

# Define the regional case we're looking at
case = 'west'

# Flag for whether we are simulating even proportions at each wealth quintile
even = False

# Define total households in the village (1 household = 1 user)
n_users_total = 200

# Define start of data path (up to quintile number)
path = 'D:\\MICS\\quintiles\\'+ case + '\\combs_q'

# PROCESS #

# Set sunrise and sunset variables for case
sunrise = sunrise_dict[case]
sunset = sunset_dict[case]

# Instantiate user list
User_list = []

# Instantiate combs dictionary list
Combs_list = []

# For each quintile
for q in range(1, 6):
    # If the even flag is on, actually get this based on even not case.
    if even:
        n_users = round(n_users_total * quintiles[q]['even'])
    else:
        # Get the amount of households in the village at that quintile
        n_users = round(n_users_total * quintiles[q][case])
    # Get the data for that quintile
    data = path + str(q) + '_' + case + '.csv'
    # Read dicts of relative prevalence of appliance combinations in that quintile
    prevalence = {}
    for row in csv.DictReader(open(data)):
        prevalence[row['comb']] = row['percent']
    # Create a placeholder for number of households per combination in the quintile
    n_per_comb = {}
    # For each comb, get the number of users in the quintile
    for key in prevalence:
        n_per_comb[key] = round(n_users * float(prevalence[key]))
    # Append the combinations for this tier to the list of dictionaries
    Combs_list.append(n_per_comb)

# In the combs list, sum the values in the dictionaries with same keys to produce full combs list
Combs_dict = dict(functools.reduce(operator.add, map(collections.Counter, Combs_list)))

print("Number of users to model: " + str(n_users_total))
print("Total households allocated to usertypes: " + str(sum(Combs_dict.values())))
print("Redistribute the difference randomly...")

adjustment = n_users_total - sum(Combs_dict.values())

# If rounding error makes the total greater than the defined total
if (sum(Combs_dict.values()) < n_users_total) or (sum(Combs_dict.values()) > n_users_total):
    # Add users randomly to types until gap is 0
    i = abs(adjustment)
    while i > 0:
        # Choose a random combination
        c = random.choice(list(Combs_dict.keys()))
        # If the adjustment is positive, add to that random combination
        if adjustment > 0:
            Combs_dict[c] = Combs_dict[c] + 1
            print("Added 1 to " + str(c))
        # If the adjustment is negative, subtract from that random combination
        if adjustment < 0:
            # Check to ensure this won't make the combination negative
            if Combs_dict[c] == 0:
                continue  # It will try again with a different combination, i doesn't increment.
            # if not, remove 1 from that random combination
            Combs_dict[c] = Combs_dict[c] - 1
            print("Removed 1 from " + str(c))
        i = i-1

# Get the usertypes as a list
usertypes = list(Combs_dict.keys())

# Define usertypes
for name in usertypes:
    # Create usertype
    C = User(name=name, n_users=Combs_dict[name])
    # Add the lights that everyone has
    C_Light_Task = C.Appliance(C, n=4, P=1, w=1, t=60, r_t=0.2, c=10)
    C_Light_Task.windows(w1=[sunset, 1440], r_w=0.35)
    # Add other appliances as needed based on combination input files:
    if "L" in C.user_name:
        C_Light_Indoors = C.Appliance(C, n=5, P=12, w=1, t=120, r_t=0.2, c=10)
        C_Light_Indoors.windows(w1=[sunset, 1440], r_w=0.35)
    if "O" in C.user_name:
        C_Light_Outdoors = C.Appliance(C, n=4, P=12, w=2, t=540, r_t=0.2, c=30, fixed='yes')
        C_Light_Outdoors.windows(w1=[sunset, 1440], w2=[0, sunrise], r_w=0.2)
    if "M" in C.user_name:
        C_Mobile_charger = C.Appliance(C, n=1, P=2, w=2, t=120, r_t=0.2, c=20)
        C_Mobile_charger.windows(w1=[1320, 1440], w2=[0, 120], r_w=0.35)
    if "R" in C.user_name:
        C_Radio = C.Appliance(C, n=1, P=4, w=1, t=90, r_t=0.2, c=30, occasional_use=0.8)
        C_Radio.windows(w1=[sunset, 1440], r_w=0.35)
    if "T" in C.user_name:
        C_TV = C.Appliance(C, n=1, P=40, w=1, t=90, r_t=0.2, c=30, occasional_use=0.8)
        C_TV.windows(w1=[sunset, 1440], r_w=0.35)
    if "F" in C.user_name:
        # Weekday fan - Window is 6PM - sunrise, 12 hours use, at least one hour after turn on.
        C_Fan_Weekday = C.Appliance(C, n=1, P=40, w=2, t=720, r_t=0.2, c=60, wd_we_type=0)
        C_Fan_Weekday.windows(w1=[1080, 1440], w2=[0, sunrise], r_w=0.2)
        # Weekend fan - same as weekday and add 12-2 so 14 hours use
        C_Fan_Weekend = C.Appliance(C, n=1, P=40, w=3, t=840, r_t=0.2, c=60, wd_we_type=1)
        C_Fan_Weekend.windows(w1=[1080, 1440], w2=[0, sunrise], w3=[720, 840], r_w=0.2)
    if "C" in C.user_name:
        C_Computer = C.Appliance(C, n=1, P=50, w=1, t=90, r_t=0.2, c=30, occasional_use=0.8)
        C_Computer.windows(w1=[sunset, 1440], r_w=0.35)
    if "I" in C.user_name:
        C_Iron = C.Appliance(C, n=1, P=1100, w=1, t=30, r_t=0.2, c=5, wd_we_type=1, thermal_P_var=0.3, occasional_use=0.8)
        C_Iron.windows(w1=[sunset, 1440], r_w=0.35)
    if "Z" in C.user_name:
        C_Fridge = C.Appliance(C, n=1, P=300, w=1, t=1440, r_t=0, fixed='yes', fixed_cycle=2)
        C_Fridge.windows(w1=[0, 1440])
        C_Fridge.specific_cycle_1(P_11=300, t_11=20, P_12=5, t_12=10)  # intensive duty cycle - sunrise to sunset
        C_Fridge.specific_cycle_2(P_21=300, t_21=10, P_22=5, t_22=20)  # light duty cycle - sunset to sunrise
        C_Fridge.cycle_behaviour(cw11=[sunrise, sunset - 1],
                                 cw21=[sunset, 1439], cw22=[0, sunrise - 1])
    User_list.append(C)




    # adjustment = n_users_total - sum(Combs_dict.values())
    # print("Original only-lights count: " + str(Combs_dict['NIO']))
    # Combs_dict['NIO'] = Combs_dict['NIO'] + adjustment
    # print("Adjusted only-lights count: " + str(Combs_dict['NIO']))
    # # If this makes the number of None households negative
    # if Combs_dict['NIO'] < 0:
    #     print("Can't remove enough None households for count correction, removing M households.")
    #     # Subtract the difference from the mobile households
    #     Combs_dict['MIO'] = Combs_dict['MIO'] + Combs_dict['NIO']
    #     # Then set to 0
    #     Combs_dict['NIO'] = 0
    #     # If this then makes the count of mobiles less than 0
    #     if Combs_dict['MIO'] < 0:
    #         # Print an error message. Could instead do nested ifs of additional appliance combs.
    #         print("Can't remove enough M households for count correction, error!")



#
    # print(n_per_comb)
    # print(sum(n_per_comb.values()))
    # # If rounding error makes the total less than or greater than the defined total for the quintile
    # if (sum(n_per_comb.values()) < n_users) or (sum(n_per_comb.values()) > n_users):
    #     # Add or subtract houses with just lights (None) to make the total the defined total
    #     adjustment = n_users_total - sum(n_per_comb.values())
    #     n_per_comb['None'] = n_per_comb['None'] + adjustment
    #     # If this makes the number of None households negative
    #     if n_per_comb['None'] < 0:
    #         print("Can't remove enough None households for count correction, removing M households.")
    #         # Subtract the difference from the mobile households
    #         n_per_comb['M'] = n_per_comb['M'] + n_per_comb['None']
    #         # Then set to 0
    #         n_per_comb['None'] = 0
    #         # If this then makes the count of mobiles less than 0
    #         if n_per_comb['M'] < 0:
    #             # Print an error message. Could instead do nested ifs of additional appliance combs.
    #             print("Can't remove enough M households for count correction, error!")
    #
    # # Read in all the usertypes seperately as a list
    # df = read_csv(data)
    # usertypes = df['comb'].tolist()
    # # Define usertypes
    # for name in usertypes:
    #
    #     # Create usertype
    #     C = User(name=name + 'q' + str(q), n_users=n_per_comb[name])
    #
    #     # Add the task lights that everyone has
    #     C_Light_Task = C.Appliance(C, n=4, P=1, w=1, t=60, r_t=0.2, c=10)
    #     C_Light_Task.windows(w1=[sunset, 1440], r_w=0.35)
    #
    #     # For quintiles 2-6, add general lights (not in Q1).
    #     if q in range(2, 6):
    #         C_Light_Indoors = C.Appliance(C, n=5, P=12, w=1, t=120, r_t=0.2, c=10)
    #         C_Light_Indoors.windows(w1=[sunset, 1440], r_w=0.35)
    #         C_Light_Outdoors = C.Appliance(C, n=4, P=12, w=2, t=540, r_t=0.2, c=30, fixed='yes')
    #         C_Light_Outdoors.windows(w1=[sunset, 1440], w2=[0, sunrise], r_w=0.2)
    #
    #     # Add other appliances as needed based on combination input files:
    #     if "M" in C.user_name:
    #         C_Mobile_charger = C.Appliance(C, n=1, P=2, w=2, t=120, r_t=0.2, c=20)
    #         C_Mobile_charger.windows(w1=[1320, 1440], w2=[0, 120], r_w=0.35)
    #     if "R" in C.user_name:
    #         C_Radio = C.Appliance(C, n=1, P=4, w=1, t=90, r_t=0.2, c=30, occasional_use=0.8)
    #         C_Radio.windows(w1=[sunset, 1440], r_w=0.35)
    #     if "T" in C.user_name:
    #         C_TV = C.Appliance(C, n=1, P=40, w=1, t=90, r_t=0.2, c=30, occasional_use=0.8)
    #         C_TV.windows(w1=[sunset, 1440], r_w=0.35)
    #     if "F" in C.user_name:
    #         # Weekday fan - Window is 6PM - sunrise, 12 hours use, at least one hour after turn on.
    #         C_Fan_Weekday = C.Appliance(C, n=1, P=40, w=2, t=720, r_t=0.2, c=60, wd_we_type=0)
    #         C_Fan_Weekday.windows(w1=[1080, 1440], w2=[0, sunrise], r_w=0.2)
    #         # Weekend fan - same as weekday and add 12-2 so 14 hours use
    #         C_Fan_Weekend = C.Appliance(C, n=1, P=40, w=3, t=840, r_t=0.2, c=60, wd_we_type=1)
    #         C_Fan_Weekend.windows(w1=[1080, 1440], w2=[0, sunrise], w3=[720, 840], r_w=0.2)
    #     if "C" in C.user_name:
    #         C_Computer = C.Appliance(C, n=1, P=50, w=1, t=90, r_t=0.2, c=30, occasional_use=0.8)
    #         C_Computer.windows(w1=[sunset, 1440], r_w=0.35)
    #     if "I" in C.user_name:
    #         C_Iron = C.Appliance(C, n=1, P=1100, w=1, t=30, r_t=0.2, c=5, wd_we_type=1, thermal_P_var=0.3, occasional_use=0.8)
    #         C_Iron.windows(w1=[sunset, 1440], r_w=0.35)
    #     if "Z" in C.user_name:
    #         C_Fridge = C.Appliance(C, n=1, P=300, w=1, t=1440, r_t=0, fixed='yes', fixed_cycle=2)
    #         C_Fridge.windows(w1=[0, 1440])
    #         C_Fridge.specific_cycle_1(P_11=300, t_11=20, P_12=5, t_12=10)  # intensive duty cycle - sunrise to sunset
    #         C_Fridge.specific_cycle_2(P_21=300, t_21=10, P_22=5, t_22=20)  # light duty cycle - sunset to sunrise
    #         C_Fridge.cycle_behaviour(cw11=[sunrise, sunset - 1],
    #                                  cw21=[sunset, 1439], cw22=[0, sunrise - 1])
    #     User_list.append(C)
    #
    #
    #
    #
    #
    #



