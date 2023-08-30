'''
This example input file contains the values for all the MICS appliances based on MICS/IHS/etc, saved as C10.

Times in minutes:
0 = 12:00 AM*
60 = 1:00 AM
120 = 2:00 AM
180 = 3:00 AM
240 = 4:00 AM
300 = 5:00 AM
360 = 6:00 AM
420 = 7:00 AM
480 = 8:00 AM
540 = 9:00 AM
600 = 10:00 AM
660 = 11:00 AM
720 = 12:00 PM
780 = 1:00 PM (13:00)
840 = 2:00 PM (14:00)
900 = 3:00 PM (15:00)
960 = 4:00 PM (16:00)
1020 = 5:00 PM (17:00)
1080 = 6:00 PM (18:00)
1140 = 7:00 PM (19:00)
1200 = 8:00 PM (20:00)
1260 = 9:00 PM (21:00)
1320 = 10:00 PM (22:00)
1380 = 11:00 PM (23:00)
1440 = 12:00 AM (24:00)*

Use average sunrise and sunset for January in Katiri: 7:09 AM, 6:47PM
i.e. in minutes: 429, 1127.

Higher powers from MTF used where two provided

Assume lights are used indoors in the evening from sunset until 12:30 AM (end time as per example file).
Assume outdoor light (security) is on from sunset until sunrise.

# Author: Alycia Leonard, Energy and Power Group, University of Oxford
# Contact: alycia.leonard@eng.ox.ac.uk
'''

# -*- coding: utf-8 -*-

#%% Definition of the inputs
'''
Input data definition
'''


from core import User, np
User_list = []


# C10 = M,R,T,F,C,I,Fr

C10 = User(name="M,R,T,F,C,I,Fr", n_users=1)
User_list.append(C10)

# C10 - Mobile, radio, television, fan, iron, computer, and fridge.

# Indoor lights
C10_L = C10.Appliance(C10, n=2, P=12, w=2, t=120, r_t=0.2, c=10)
C10_L.windows(w1=[1127, 1440], w2=[0, 30], r_w=0.35)

# Outdoor light - 10 hours from sample file
C10_Lo = C10.Appliance(C10, n=1, P=12, w=2, t=600, r_t=0.2, c=10)
C10_Lo.windows(w1=[0, 429], w2=[1127, 1440], r_w=0.35)

# Mobile phone charger - t = 240 from MTF table 6.13, 1020 from samples
C10_M = C10.Appliance(C10, n=1, P=2, w=1, t=240, r_t=0.2, c=5)
C10_M.windows(w1=[1020, 1440], w2=[0, 0], w3=[0, 0], r_w=0.35)

# Radio - assumed turned on at sunrise for an hour, sunset for two hours from samples
C10_R = C10.Appliance(C10, n=1, P=4, w=2, t=60, r_t=0.1, c=5)
C10_R.windows(w1=[429, 489], w2=[1127, 1247], r_w=0.35)

# Television - assume window after dinner at 7:30 until 12:30 am
C10_T = C10.Appliance(C10, n=1, P=40, w=2, t=90, r_t=0.1, c=5)
C10_T.windows(w1=[1170, 1440], w2=[0, 30], r_w=0.35)

# Fan - assume on from sunrise for an hour, sunset for two hours from sample
C10_F = C10.Appliance(C10, n=1, P=40, w=2, t=60, r_t=0.1, c=5)
C10_F.windows(w1=[429, 489], w2=[1127, 1247], r_w=0.35)

# Computer - 50 assumed from google search. Use from 7:30 PM to 12:30 AM.
C10_C = C10.Appliance(C10, n=1, P=50, w=2, t=60, r_t=0.1, c=5)
C10_C.windows(w1=[1170, 1440], w2=[0, 30], r_w=0.35)

# Iron - assume used occasionally in the morning or before evening.
C10_I = C10.Appliance(C10, n=1, P=1100, w=2, t=30, r_t=0.1, c=1, occasional_use=0.33)
C10_I.windows(w1=[429, 459], w2=[1020, 1170], r_w=0.35)

# Fridge/freezer - duty cycles left as they were in example, only powers changed.
C10_Fr = C10.Appliance(C10, n=1, P=300, w=1, t=1440, r_t=0, c=30, fixed='yes', fixed_cycle=3)
C10_Fr.windows(w1=[0, 1440], w2=[0, 0])
C10_Fr.specific_cycle_1(P_11=300, t_11=20, P_12=5, t_12=10)
C10_Fr.specific_cycle_2(P_21=300, t_21=15, P_22=5, t_22=15)
C10_Fr.specific_cycle_3(P_31=300, t_31=10, P_32=5, t_32=20)
C10_Fr.cycle_behaviour(cw11=[480, 1200], cw12=[0, 0], cw21=[300, 479], cw22=[0, 0], cw31=[0, 299], cw32=[1201, 1440])




