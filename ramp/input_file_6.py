'''
This input file represents the Mapoton case study.
Data is from Mapoton footprints, MICS North SLE, MTF powers, sunrise/sunset from NOAA.

Use average sunrise and sunset for January in Mapoton: 7:09 AM, 6:47PM
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



# Define total households in village (1 household = 1 user)
n_users_total = 133

# Define proportion of households for each usertype based on North SLE
prevalence = [0.04, 0.09, 0.03, 0.14, 0.04, 0.08, 0.10, 0.03, 0.07, 0.05, 0.04]

# Create new user classes
# User classes correspond to appliance combinations from MICS, labelled C00 through C10 (increasing access order)
# Assume all combinations have lights (indoor and outdoor). Use LI Indoor and Outdoor bulb stats from input_file_1.

# C00 = None
C00 = User(name="none", n_users=int(n_users_total*prevalence[0]))
User_list.append(C00)
# C01 = M
C01 = User(name="M", n_users=int(n_users_total*prevalence[1]))
User_list.append(C01)
# C02 = R
C02 = User(name="R", n_users=int(n_users_total*prevalence[2]))
User_list.append(C02)
# C03 = M,R
C03 = User(name="M,R", n_users=int(n_users_total*prevalence[3]))
User_list.append(C03)
# C04 = M,T
C04 = User(name="M,T", n_users=int(n_users_total*prevalence[4]))
User_list.append(C04)
# C05 = M,R,T
C05 = User(name="M,R,T", n_users=int(n_users_total*prevalence[5]))
User_list.append(C05)
# C06 = M,R,T,F
C06 = User(name="M,R,T,F", n_users=int(n_users_total*prevalence[6]))
User_list.append(C06)
# C07 = M,R,T,Fr
C07 = User(name="M,R,T,Fr", n_users=int(n_users_total*prevalence[7]))
User_list.append(C07)
# C08 = M,R,T,F,Fr
C08 = User(name="M,R,T,F,Fr", n_users=int(n_users_total*prevalence[8]))
User_list.append(C08)
# C09 = M,R,T,F,I,Fr
C09 = User(name="M,R,T,F,I,Fr", n_users=int(n_users_total*prevalence[9]))
User_list.append(C09)
# C10 = M,R,T,F,C,I,Fr
C10 = User(name="M,R,T,F,C,I,Fr", n_users=int(n_users_total*prevalence[10]))
User_list.append(C10)

# Create new appliances for each user class

# C00 - No appliances, just lights.

# Indoor lights
C00_L = C00.Appliance(C00, n=2, P=12, w=2, t=120, r_t=0.2, c=10)
C00_L.windows(w1=[1127, 1440], w2=[0, 30], r_w=0.35)

# Outdoor light - 10 hours from sample file
C00_Lo = C00.Appliance(C00, n=1, P=12, w=2, t=600, r_t=0.2, c=10)
C00_Lo.windows(w1=[0, 429], w2=[1127, 1440], r_w=0.35)

# C01 - Mobile only.

# Indoor lights
C01_L = C01.Appliance(C01, n=2, P=12, w=2, t=120, r_t=0.2, c=10)
C01_L.windows(w1=[1127, 1440], w2=[0, 30], r_w=0.35)

# Outdoor light - 10 hours from sample file
C01_Lo = C01.Appliance(C01, n=1, P=12, w=2, t=600, r_t=0.2, c=10)
C01_Lo.windows(w1=[0, 429], w2=[1127, 1440], r_w=0.35)

# Mobile phone charger - t = 240 from MTF table 6.13, 1020 from samples
C01_M = C01.Appliance(C01, n=1, P=2, w=1, t=240, r_t=0.2, c=5)
C01_M.windows(w1=[1020, 1440], w2=[0, 0], w3=[0, 0], r_w=0.35)


# C02 - Radio only.

# Indoor lights
C02_L = C02.Appliance(C02, n=2, P=12, w=2, t=120, r_t=0.2, c=10)
C02_L.windows(w1=[1127, 1440], w2=[0, 30], r_w=0.35)

# Outdoor light - 10 hours from sample file
C02_Lo = C02.Appliance(C02, n=1, P=12, w=2, t=600, r_t=0.2, c=10)
C02_Lo.windows(w1=[0, 429], w2=[1127, 1440], r_w=0.35)

# Radio - assumed turned on at sunrise for an hour, sunset for two hours from samples
C02_R = C02.Appliance(C02, n=1, P=4, w=2, t=60, r_t=0.1, c=5)
C02_R.windows(w1=[429, 489], w2=[1127, 1247], r_w=0.35)

# C03 - Mobile and Radio.

# Indoor lights
C03_L = C03.Appliance(C03, n=2, P=12, w=2, t=120, r_t=0.2, c=10)
C03_L.windows(w1=[1127, 1440], w2=[0, 30], r_w=0.35)

# Outdoor light - 10 hours from sample file
C03_Lo = C03.Appliance(C03, n=1, P=12, w=2, t=600, r_t=0.2, c=10)
C03_Lo.windows(w1=[0, 429], w2=[1127, 1440], r_w=0.35)

# Mobile phone charger - t = 240 from MTF table 6.13, 1020 from samples
C03_M = C03.Appliance(C03, n=1, P=2, w=1, t=240, r_t=0.2, c=5)
C03_M.windows(w1=[1020, 1440], w2=[0, 0], w3=[0, 0], r_w=0.35)

# Radio - assumed turned on at sunrise for an hour, sunset for two hours from samples
C03_R = C03.Appliance(C03, n=1, P=4, w=2, t=60, r_t=0.1, c=5)
C03_R.windows(w1=[429, 489], w2=[1127, 1247], r_w=0.35)

# C04 - Mobile and Television.

# Indoor lights
C04_L = C04.Appliance(C04, n=2, P=12, w=2, t=120, r_t=0.2, c=10)
C04_L.windows(w1=[1127, 1440], w2=[0, 30], r_w=0.35)

# Outdoor light - 10 hours from sample file
C04_Lo = C04.Appliance(C04, n=1, P=12, w=2, t=600, r_t=0.2, c=10)
C04_Lo.windows(w1=[0, 429], w2=[1127, 1440], r_w=0.35)

# Mobile phone charger - t = 240 from MTF table 6.13, 1020 from samples
C04_M = C04.Appliance(C04, n=1, P=2, w=1, t=240, r_t=0.2, c=5)
C04_M.windows(w1=[1020, 1440], w2=[0, 0], w3=[0, 0], r_w=0.35)

# Television - assume window after dinner at 7:30 until 12:30 am
C04_T = C04.Appliance(C04, n=1, P=40, w=2, t=90, r_t=0.1, c=5)
C04_T.windows(w1=[1170, 1440], w2=[0, 30], r_w=0.35)

# C05 - Mobile, radio, and television.

# Indoor lights
C05_L = C05.Appliance(C05, n=2, P=12, w=2, t=120, r_t=0.2, c=10)
C05_L.windows(w1=[1127, 1440], w2=[0, 30], r_w=0.35)

# Outdoor light - 10 hours from sample file
C05_Lo = C05.Appliance(C05, n=1, P=12, w=2, t=600, r_t=0.2, c=10)
C05_Lo.windows(w1=[0, 429], w2=[1127, 1440], r_w=0.35)

# Mobile phone charger - t = 240 from MTF table 6.13, 1020 from samples
C05_M = C05.Appliance(C05, n=1, P=2, w=1, t=240, r_t=0.2, c=5)
C05_M.windows(w1=[1020, 1440], w2=[0, 0], w3=[0, 0], r_w=0.35)

# Radio - assumed turned on at sunrise for an hour, sunset for two hours from samples
C05_R = C05.Appliance(C05, n=1, P=4, w=2, t=60, r_t=0.1, c=5)
C05_R.windows(w1=[429, 489], w2=[1127, 1247], r_w=0.35)

# Television - assume window after dinner at 7:30 until 12:30 am
C05_T = C05.Appliance(C05, n=1, P=40, w=2, t=90, r_t=0.1, c=5)
C05_T.windows(w1=[1170, 1440], w2=[0, 30], r_w=0.35)

# C06 - Mobile, radio, television, and fan.

# Indoor lights
C06_L = C06.Appliance(C06, n=2, P=12, w=2, t=120, r_t=0.2, c=10)
C06_L.windows(w1=[1127, 1440], w2=[0, 30], r_w=0.35)

# Outdoor light - 10 hours from sample file
C06_Lo = C06.Appliance(C06, n=1, P=12, w=2, t=600, r_t=0.2, c=10)
C06_Lo.windows(w1=[0, 429], w2=[1127, 1440], r_w=0.35)

# Mobile phone charger - t = 240 from MTF table 6.13, 1020 from samples
C06_M = C06.Appliance(C06, n=1, P=2, w=1, t=240, r_t=0.2, c=5)
C06_M.windows(w1=[1020, 1440], w2=[0, 0], w3=[0, 0], r_w=0.35)

# Radio - assumed turned on at sunrise for an hour, sunset for two hours from samples
C06_R = C06.Appliance(C06, n=1, P=4, w=2, t=60, r_t=0.1, c=5)
C06_R.windows(w1=[429, 489], w2=[1127, 1247], r_w=0.35)

# Television - assume window after dinner at 7:30 until 12:30 am
C06_T = C06.Appliance(C06, n=1, P=40, w=2, t=90, r_t=0.1, c=5)
C06_T.windows(w1=[1170, 1440], w2=[0, 30], r_w=0.35)

# Fan - assume on from sunrise for an hour, sunset for two hours from sample
C06_F = C06.Appliance(C06, n=1, P=40, w=2, t=60, r_t=0.1, c=5)
C06_F.windows(w1=[429, 489], w2=[1127, 1247], r_w=0.35)

# C07 - Mobile, radio, television, and fridge.

# Indoor lights
C07_L = C07.Appliance(C07, n=2, P=12, w=2, t=120, r_t=0.2, c=10)
C07_L.windows(w1=[1127, 1440], w2=[0, 30], r_w=0.35)

# Outdoor light - 10 hours from sample file
C07_Lo = C07.Appliance(C07, n=1, P=12, w=2, t=600, r_t=0.2, c=10)
C07_Lo.windows(w1=[0, 429], w2=[1127, 1440], r_w=0.35)

# Mobile phone charger - t = 240 from MTF table 6.13, 1020 from samples
C07_M = C07.Appliance(C07, n=1, P=2, w=1, t=240, r_t=0.2, c=5)
C07_M.windows(w1=[1020, 1440], w2=[0, 0], w3=[0, 0], r_w=0.35)

# Radio - assumed turned on at sunrise for an hour, sunset for two hours from samples
C07_R = C07.Appliance(C07, n=1, P=4, w=2, t=60, r_t=0.1, c=5)
C07_R.windows(w1=[429, 489], w2=[1127, 1247], r_w=0.35)

# Television - assume window after dinner at 7:30 until 12:30 am
C07_T = C07.Appliance(C07, n=1, P=40, w=2, t=90, r_t=0.1, c=5)
C07_T.windows(w1=[1170, 1440], w2=[0, 30], r_w=0.35)

# Fridge/freezer - duty cycles left as they were in example, only powers changed.
C07_Fr = C07.Appliance(C07, n=1, P=300, w=1, t=1440, r_t=0, c=30, fixed='yes', fixed_cycle=3)
C07_Fr.windows(w1=[0, 1440], w2=[0, 0])
C07_Fr.specific_cycle_1(P_11=300, t_11=20, P_12=5, t_12=10)
C07_Fr.specific_cycle_2(P_21=300, t_21=15, P_22=5, t_22=15)
C07_Fr.specific_cycle_3(P_31=300, t_31=10, P_32=5, t_32=20)
C07_Fr.cycle_behaviour(cw11=[480, 1200], cw12=[0, 0], cw21=[300, 479], cw22=[0, 0], cw31=[0, 299], cw32=[1201, 1440])

# C08 - Mobile, radio, television, fan, and fridge.

# Indoor lights
C08_L = C08.Appliance(C08, n=2, P=12, w=2, t=120, r_t=0.2, c=10)
C08_L.windows(w1=[1127, 1440], w2=[0, 30], r_w=0.35)

# Outdoor light - 10 hours from sample file
C08_Lo = C08.Appliance(C08, n=1, P=12, w=2, t=600, r_t=0.2, c=10)
C08_Lo.windows(w1=[0, 429], w2=[1127, 1440], r_w=0.35)

# Mobile phone charger - t = 240 from MTF table 6.13, 1020 from samples
C08_M = C08.Appliance(C08, n=1, P=2, w=1, t=240, r_t=0.2, c=5)
C08_M.windows(w1=[1020, 1440], w2=[0, 0], w3=[0, 0], r_w=0.35)

# Radio - assumed turned on at sunrise for an hour, sunset for two hours from samples
C08_R = C08.Appliance(C08, n=1, P=4, w=2, t=60, r_t=0.1, c=5)
C08_R.windows(w1=[429, 489], w2=[1127, 1247], r_w=0.35)

# Television - assume window after dinner at 7:30 until 12:30 am
C08_T = C08.Appliance(C08, n=1, P=40, w=2, t=90, r_t=0.1, c=5)
C08_T.windows(w1=[1170, 1440], w2=[0, 30], r_w=0.35)

# Fan - assume on from sunrise for an hour, sunset for two hours from sample
C08_F = C08.Appliance(C08, n=1, P=40, w=2, t=60, r_t=0.1, c=5)
C08_F.windows(w1=[429, 489], w2=[1127, 1247], r_w=0.35)

# Fridge/freezer - duty cycles left as they were in example, only powers changed.
C08_Fr = C08.Appliance(C08, n=1, P=300, w=1, t=1440, r_t=0, c=30, fixed='yes', fixed_cycle=3)
C08_Fr.windows(w1=[0, 1440], w2=[0, 0])
C08_Fr.specific_cycle_1(P_11=300, t_11=20, P_12=5, t_12=10)
C08_Fr.specific_cycle_2(P_21=300, t_21=15, P_22=5, t_22=15)
C08_Fr.specific_cycle_3(P_31=300, t_31=10, P_32=5, t_32=20)
C08_Fr.cycle_behaviour(cw11=[480, 1200], cw12=[0, 0], cw21=[300, 479], cw22=[0, 0], cw31=[0, 299], cw32=[1201, 1440])

# C09 - Mobile, radio, television, fan, iron, and fridge.

# Indoor lights
C09_L = C09.Appliance(C09, n=2, P=12, w=2, t=120, r_t=0.2, c=10)
C09_L.windows(w1=[1127, 1440], w2=[0, 30], r_w=0.35)

# Outdoor light - 10 hours from sample file
C09_Lo = C09.Appliance(C09, n=1, P=12, w=2, t=600, r_t=0.2, c=10)
C09_Lo.windows(w1=[0, 429], w2=[1127, 1440], r_w=0.35)

# Mobile phone charger - t = 240 from MTF table 6.13, 1020 from samples
C09_M = C09.Appliance(C09, n=1, P=2, w=1, t=240, r_t=0.2, c=5)
C09_M.windows(w1=[1020, 1440], w2=[0, 0], w3=[0, 0], r_w=0.35)

# Radio - assumed turned on at sunrise for an hour, sunset for two hours from samples
C09_R = C09.Appliance(C09, n=1, P=4, w=2, t=60, r_t=0.1, c=5)
C09_R.windows(w1=[429, 489], w2=[1127, 1247], r_w=0.35)

# Television - assume window after dinner at 7:30 until 12:30 am
C09_T = C09.Appliance(C09, n=1, P=40, w=2, t=90, r_t=0.1, c=5)
C09_T.windows(w1=[1170, 1440], w2=[0, 30], r_w=0.35)

# Fan - assume on from sunrise for an hour, sunset for two hours from sample
C09_F = C09.Appliance(C09, n=1, P=40, w=2, t=60, r_t=0.1, c=5)
C09_F.windows(w1=[429, 489], w2=[1127, 1247], r_w=0.35)

# Iron - assume used occasionally in the morning or before evening.
C09_I = C09.Appliance(C09, n=1, P=1100, w=2, t=30, r_t=0.1, c=1, occasional_use=0.33)
C09_I.windows(w1=[429, 459], w2=[1020, 1170], r_w=0.35)

# Fridge/freezer - duty cycles left as they were in example, only powers changed.
C09_Fr = C09.Appliance(C09, n=1, P=300, w=1, t=1440, r_t=0, c=30, fixed='yes', fixed_cycle=3)
C09_Fr.windows(w1=[0, 1440], w2=[0, 0])
C09_Fr.specific_cycle_1(P_11=300, t_11=20, P_12=5, t_12=10)
C09_Fr.specific_cycle_2(P_21=300, t_21=15, P_22=5, t_22=15)
C09_Fr.specific_cycle_3(P_31=300, t_31=10, P_32=5, t_32=20)
C09_Fr.cycle_behaviour(cw11=[480, 1200], cw12=[0, 0], cw21=[300, 479], cw22=[0, 0], cw31=[0, 299], cw32=[1201, 1440])

# C10 - Mobile, radio, television, fan, computer, iron, and fridge.

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
