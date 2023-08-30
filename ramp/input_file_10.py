'''
This input file constructs a profile for the Mapoton case study village for a given MTF tier.
Tiers 1, 2, and 4 (given the appliances included in the normal setup)
Tier 1: Phone and radio
Tier 2: Phone, radio, TV, fan, computer
Tier 4: Phone, radio, TV, fan, computer, iron, fridge
Results saved in v3 eval folder under tier_results_stochastic or similar
# Author: Alycia Leonard, Energy and Power Group, University of Oxford
# Contact: alycia.leonard@eng.ox.ac.uk
'''

from core import *

# INPUT DATA #

tier = 4

# Define the case we're looking at
case = "overall"

# Define total households in synthetic village (1 household = 1 user)
n_users_total = 200

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

# Instantiate user list
User_list = []

C = User(name='tier', n_users=n_users_total)

if tier == 1:
    # Tier 1 "task lighting" rated at 1 W per unit for 4 hours a day.
    # reduced to 4 bulbs, time of use reduced to 60 per bulb
    C_Light_Indoors = C.Appliance(C, n=4, P=1, w=1, t=60, r_t=0.2, c=10)
    C_Light_Indoors.windows(w1=[sunset, 1440], r_w=0.35)

    # Tier 1 phone charging at 2 W for 2 hours a day
    C_Mobile_charger = C.Appliance(C, n=1, P=2, w=2, t=120, r_t=0.2, c=20)
    C_Mobile_charger.windows(w1=[1320, 1440], w2=[0, 120], r_w=0.35)

    # Tier 1 radio is 2 W for 2 hours a day. Get rid of occasional use.
    # This is likely to be battery radio, but ignore that.
    C_Radio = C.Appliance(C, n=1, P=2, w=1, t=120, r_t=0.2, c=30)
    C_Radio.windows(w1=[sunset, 1440], r_w=0.35)

if tier == 2:
    # Tier 2 "task lighting" rated at 2 W per unit for 4 hours a day.
    # reduced to 4 bulbs, time of use reduced to 60 per bulb
    C_Light_Indoors = C.Appliance(C, n=4, P=2, w=1, t=60, r_t=0.2, c=10)
    C_Light_Indoors.windows(w1=[sunset, 1440], r_w=0.35)

    # Tier 2 general lighting - 12 W per unit, 4 hours a day.
    # Reduce to 1 hour per bulb with 4 bulbs to achieve this.
    C_Light_Outdoors = C.Appliance(C, n=4, P=12, w=2, t=60, r_t=0.2, c=30) # , fixed='yes')
    C_Light_Outdoors.windows(w1=[sunset, 1440], w2=[0, sunrise], r_w=0.2)

    # Tier 2 phone charging - 2 W, 4 hours a day. Extend window to 4 AM
    C_Mobile_charger = C.Appliance(C, n=1, P=2, w=2, t=240, r_t=0.2, c=20)
    C_Mobile_charger.windows(w1=[1320, 1440], w2=[0, 240], r_w=0.35)

    # Tier 2 radio - 4 W, 4 hours a day. Get rid of occasional use.
    # This is likely to be battery radio, but ignore that.
    C_Radio = C.Appliance(C, n=1, P=4, w=1, t=240, r_t=0.2, c=30)
    C_Radio.windows(w1=[sunset, 1440], r_w=0.35)

    # Tier 2 television - 20 W, 2 hours. Remove occasional use.
    C_TV = C.Appliance(C, n=1, P=20, w=1, t=120, r_t=0.2, c=30)
    C_TV.windows(w1=[sunset, 1440], r_w=0.35)

    # Tier 2 fan - 4 hours per day and 20 W so do that at night, no weekend variability.
    C_Fan = C.Appliance(C, n=1, P=20, w=2, t=240, r_t=0.2, c=60)
    C_Fan.windows(w1=[1080, 1440], w2=[0, sunrise], r_w=0.2)

    # Computer not defined in 6.13 - leave as is but remove occasional use.
    C_Computer = C.Appliance(C, n=1, P=50, w=1, t=90, r_t=0.2, c=30)
    C_Computer.windows(w1=[sunset, 1440], r_w=0.35)

if tier == 4:
    # Tier 4 "task lighting" rated at 2 W per unit for 8 hours a day.
    # reduced to 4 bulbs, time of use reduced to 120 per bulb
    C_Light_Indoors = C.Appliance(C, n=4, P=2, w=1, t=120, r_t=0.2, c=10)
    C_Light_Indoors.windows(w1=[sunset, 1440], r_w=0.35)

    # Tier 4 general lighting - 12 W per unit, 8 hours a day.
    # Reduce to 2 hour per bulb with 4 bulbs to achieve this.
    C_Light_Outdoors = C.Appliance(C, n=4, P=12, w=2, t=120, r_t=0.2, c=30)  # , fixed='yes')
    C_Light_Outdoors.windows(w1=[sunset, 1440], w2=[0, sunrise], r_w=0.2)

    # Tier 4 phone charging - 2 W, 4 hours a day. Extend window to 4 AM
    C_Mobile_charger = C.Appliance(C, n=1, P=2, w=2, t=240, r_t=0.2, c=20)
    C_Mobile_charger.windows(w1=[1320, 1440], w2=[0, 240], r_w=0.35)

    # Tier 4 radio - 4 W, 4 hours a day. Get rid of occasional use.
    # This is likely to be battery radio, but ignore that.
    C_Radio = C.Appliance(C, n=1, P=4, w=1, t=240, r_t=0.2, c=30)
    C_Radio.windows(w1=[sunset, 1440], r_w=0.35)

    # Tier 4 television - 40 W, 2 hours. Remove occasional use.
    C_TV = C.Appliance(C, n=1, P=40, w=1, t=120, r_t=0.2, c=30)
    C_TV.windows(w1=[sunset, 1440], r_w=0.35)

    # Tier 4 fan - 12 hours per day and 40 W so do that at night, no weekend variability.
    C_Fan = C.Appliance(C, n=1, P=40, w=2, t=720, r_t=0.2, c=60)
    C_Fan.windows(w1=[1080, 1440], w2=[0, sunrise], r_w=0.2)

    # Computer not defined in 6.13 - leave as is but remove occasional use.
    C_Computer = C.Appliance(C, n=1, P=50, w=1, t=90, r_t=0.2, c=30)
    C_Computer.windows(w1=[sunset, 1440], r_w=0.35)

    # Tier 4 iron - 1100 wats and 20 minutes use per day
    # No weekend or weekday definition, no power variability, no occasional use.
    C_Iron = C.Appliance(C, n=1, P=1100, w=1, t=20, r_t=0.2, c=5)
    C_Iron.windows(w1=[sunset, 1440], r_w=0.35)

    # Tier 4 refrigerator - 300 watts, 6 hours use per day. How do you only turn on a fridge for 6 hours a day.
    # No duty cycling. Oh boy. So 6 hours from sunrise to sunset i guess, assume used during day?
    # Assume turned on for all 6 hours in a row.
    # Add r_t and r_w are 0.2 since not used all day now.
    #C_Fridge = C.Appliance(C, n=1, P=300, w=1, t=360, r_t=0.2, c=360)
    #C_Fridge.windows(w1=[sunrise, sunset], r_w=0.2)

    # Alternative fridge: Interpret 6 hours a day of use as a 1/4 time on duty cycle
    C_Fridge = C.Appliance(C, n=1, P=300, w=1, t=1440, r_t=0, fixed='yes', fixed_cycle=1)
    C_Fridge.windows(w1=[0, 1440])
    C_Fridge.specific_cycle_1(P_11=300, t_11=5, P_12=0, t_12=15)  # On 5 minutes, off 15 = 1/4 on duty cycle all day
    C_Fridge.cycle_behaviour(cw11=[0, 1440])

User_list.append(C)



