'''
This input file represents the synthetic community case study for the four regions and the Mapoton case study.
Higher powers from MTF used where two provided
Alycia Leonard 2021-01-15 alycia.leonard@eng.ox.ac.uk
Corresponds with results in "v3" folder
'''

import csv
from core import *
from pandas import *

# INPUT DATA #

# Define the case we're looking at
case = "north"

# Define total households in synthetic village (1 household = 1 user)
n_users_total = 133

# Define sunrise and sunset times in each case
sunrise_dict = {
    'north': 426,
    'south': 424,
    'east': 422,
    'west': 432,
    'urban': 426,
    'rural': 426,
    'overall': 426,
    'mapoton': 429
}

sunset_dict = {
    'north': 1125,
    'south': 1129,
    'east': 1126,
    'west': 1133,
    'urban': 1128,
    'rural': 1128,
    'overall': 1128,
    'mapoton': 1127
}

# PROCESS #

# Set sunrise and sunset variables for case
sunrise = sunrise_dict[case]
sunset = sunset_dict[case]

# Path to data file for region considered
data = 'D:\\MICS\\' + case + '.csv'

# Read dicts of relative prevalence for region considered
reader = csv.DictReader(open(data))
prev = {}
for row in reader:
    prev[row['comb']] = row['percent']

# Read in the usertypes seperately as a list
df = read_csv(data)
usertypes = df['comb'].tolist()

# Placeholder for number of households per combination
n_per_comb = {}

# For each comb, get the number of users
for key in prev:
    n_per_comb[key] = round(n_users_total * float(prev[key]))

# If rounding error makes the total less than or greater than the defined total
if (sum(n_per_comb.values()) < n_users_total) or (sum(n_per_comb.values()) > n_users_total):
    # Add or subtract houses with just lights (None) to make the total the defined total
    adjustment = n_users_total - sum(n_per_comb.values())
    n_per_comb['None'] = n_per_comb['None'] + adjustment

# Instantiate user list
User_list = []

# Define each usertype
for name in usertypes:

    C = User(name=name, n_users=n_per_comb[name])

    C_Light_Indoors = C.Appliance(C, n=5, P=12, w=1, t=120, r_t=0.2, c=10)
    C_Light_Indoors.windows(w1=[sunset, 1440], r_w=0.35)

    C_Light_Outdoors = C.Appliance(C, n=4, P=12, w=2, t=540, r_t=0.2, c=30, fixed='yes')
    C_Light_Outdoors.windows(w1=[sunset, 1440], w2=[0, sunrise], r_w=0.2)

    # Add other appliances as needed based on combination:

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






# Add indoor lights - Window is sunset to midnight.
# Time of use is 2 hours (per bulb). Mininum ten minutes on after turn-on event.
# Add outdoor lights - Window is sunset to sunrise.
# Time of use is 12 hours. Minimum use of 10 hours after switch on. Fixed - all switch on at same time.
# Mobile phone charger - Window 10PM-midnight, two hour charge time, charge at least half an hour after connecting.
# Radio - Window is sunset to midnight, 1.5 hour use, minimum use 30 mins after switch-on.
# TV - Window is sunset to midnight, 1.5 hour use, minimum use 30 mins after switch-on.
# Computer - Window is sunset to midnight, 1.5 hour use, minimum use 30 mins after switch-on.
# Fan - Different weekend and weekday behaviour. Define one for weekday, one for weekend.
# Iron - Weekend-only appliance, evening use (sunset to midnight) for 30 mins.
# Likely to only use one of the days, so occasional at 50%.
# Ironing church or work clothes. Also is thermal, P varies by 30%.



