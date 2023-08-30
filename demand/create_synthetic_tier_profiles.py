# Create synthetic demand profiles along MTF tiers from Narayan data
# Data source: https://doi.org/10.4121/uuid:c8efa325-87fe-4125-961e-9f2684cd2086
# Author: Alycia Leonard, Energy and Power Group, University of Oxford
# Contact: alycia.leonard@eng.ox.ac.uk

import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import random

######### LOAD DEMAND DATA #################

# Define demand data path
data_path = "C:\\Users\\alyci\\Documents\\DPhil\\Demand\\Tools\\loadprofiles_MTF_dataset\\data_loadprofile_MTF_Tier5.mat"

# Load demand data into python as a dict
mat = scipy.io.loadmat(data_path)

######### SAVE PROFILE TO CSV ################

# Define save location
save_path = "C:\\Users\\alyci\\Documents\\DPhil\\Working\\Demand\\loadprofile_T5_yearly.csv"

# Save the loadprofile_T1_yearly data as CSV
np.savetxt(save_path, mat["loadprofile_T5_yearly"][0], delimiter=",")

########## MAKE RANDOM PROFILES ##############3

# Get standard yearly profile
std_yearly_profile = mat["loadprofile_T5_yearly"][0]

# Set number n of random profiles you'd like to get
n = 10

# Make n random profiles
for j in range(n):
    # Define save location
    save_path = "C:\\Users\\alyci\\Documents\\DPhil\\Working\\Demand\\T5_rand\\loadprofile_T5_random_" + str(j) + ".csv"

    # Create empty list for random yearly profile.
    rand_yearly_profile = []

    # For each day of the new profile
    for i in range(365): # 0-364
        # Get a random day in the year from the standard profile
        day = random.randint(0, 364)
        # Get the values of that day to the random yearly profile
        day_values = std_yearly_profile[(day*1440):(day*1440+1440)]
        # Turn those into a list
        day_values_list = day_values.tolist()
        # Add those values onto the random profile
        rand_yearly_profile.extend(day_values_list)

    # Convert back to numpy array to save as CSV
    rand_yearly_profile_arr = np.array(rand_yearly_profile)
    np.savetxt(save_path, rand_yearly_profile_arr, delimiter=",")

    # print(rand_yearly_profile)
    # print(len(rand_yearly_profile))

############# GET AVERAGE DAILY PROFILE FROM YEARLY DATA ##############

# Get yearly from CSV
yearly = np.genfromtxt('C:\\Users\\alyci\\Documents\\DPhil\\Working\\Demand\\T5_rand\\loadprofile_T5_random_combined_np.csv',
                        delimiter=',')

# Create list to hold average
avg = []

# For each minute of the day
for i in range(1440):
    # Create a dummy variable to hold power for that minute
    p = 0
    # For each day in the yearly profile
    for j in range(365):
        # Add the power for that minute to the dummy variable
        p = p + yearly[(1440*j)+i]
    # Take the average of the sum p
    p = p/365
    # Append p to the list
    avg.append(p)

# Save average daily load profile
avg_np = np.array(avg)
np.savetxt("C:\\Users\\alyci\\Documents\\DPhil\\Working\\Demand\\T5_rand\\loadprofile_T5_random_avg.csv", avg_np, delimiter=",")

# Plot average daily profile
plt.scatter(np.arange(1440).tolist(), avg, s=16, color='black')
plt.title("T5 average daily load profile (over n=10 yearly random T5 profiles)")
plt.xlabel('Time (minutes)')
plt.ylabel('Power (W)')
plt.show()

######### SAVE EACH DAY FROM YEARLY SEPERATELY ############

# Path to save days
days_save_path = "C:\\Users\\alyci\\Documents\\DPhil\\Working\\Demand\\Days\\T4_days\\T4_"

# Get standard yearly profile
std_yearly_profile = mat["loadprofile_T5_yearly"][0]

# Make a list to hold days
days = []

# For each day in the yearly dataset
for i in range(365):
    # Get the values for that day
    day_values = std_yearly_profile[(i*1440):(i*1440 + 1440)]
    # Append to list of days
    days.append(day_values.tolist())
    # Save day values as separate file
    np.savetxt(days_save_path + str(i) + ".csv", day_values, delimiter=",")

######## MAKE RANDOM PROFILES FROM DAYS ##############

# Number of random profiles to make
n = 1000

# Path to save random profiles
save_path = "C:\\Users\\alyci\\Documents\\DPhil\\Working\\HOMER\\Exp2_LCOE\\T5\\1000_household\\loadprofile_"

# Create numpy array in which we will sum household profiles as we go
profile_sum = np.empty(525600)

# Make random profiles from the days
for i in range(n):
    # Make a placeholder for a new random profile
    profile = []
    # For each day of the new profile of random days
    for j in range(365):  # 0-364
        # Get a random day in the year from the standard profile
        n = random.randint(0, 364)
        # Tack that day on the end of the profile
        profile.extend(days[n])
    # Save new profile as CSV
    profile_np = np.array(profile)
    np.savetxt(save_path + str(i) + ".csv", profile_np, delimiter=",")
    # Add the new profile to the sum
    profile_sum = np.add(profile_sum, profile_np)

# Save the sum of all profiles to csv
np.savetxt(save_path + "sum.csv", profile_sum, delimiter=",")


#### PLOT ONE MONTH OF EACH TIER LOAD ############

# Get yearly profiles from CSV
T1_1 = np.genfromtxt('C:\\Users\\alyci\\Documents\\DPhil\\Working\\HOMER\\Exp2_LCOE\\T1\\1_household\\loadprofile_T1_1.csv', delimiter=',')
T1_10 = np.genfromtxt('C:\\Users\\alyci\\Documents\\DPhil\\Working\\HOMER\\Exp2_LCOE\\T1\\10_household\\loadprofile_T1_10.csv', delimiter=',')
T1_100 = np.genfromtxt('C:\\Users\\alyci\\Documents\\DPhil\\Working\\HOMER\\Exp2_LCOE\\T1\\100_household\\loadprofile_T1_100.csv', delimiter=',')
T1_1000 = np.genfromtxt('C:\\Users\\alyci\\Documents\\DPhil\\Working\\HOMER\\Exp2_LCOE\\T1\\1000_household\\loadprofile_T1_1000.csv', delimiter=',')

T2 = np.genfromtxt('C:\\Users\\alyci\\Documents\\DPhil\\Working\\Demand\\T2_rand\\loadprofile_T2_random_combined_np.csv', delimiter=',')
T3 = np.genfromtxt('C:\\Users\\alyci\\Documents\\DPhil\\Working\\Demand\\T3_rand\\loadprofile_T3_random_combined_np.csv', delimiter=',')
T4 = np.genfromtxt('C:\\Users\\alyci\\Documents\\DPhil\\Working\\Demand\\T4_rand\\loadprofile_T4_random_combined_np.csv', delimiter=',')
T5 = np.genfromtxt('C:\\Users\\alyci\\Documents\\DPhil\\Working\\Demand\\T5_rand\\loadprofile_T5_random_combined_np.csv', delimiter=',')

# Plot them

fig = plt.figure(figsize=(18, 4))
ax1 = fig.add_subplot(111)
ax1.plot(np.arange(10080), T1_1[:10080], c='blue', label='T1_1')
ax1.plot(np.arange(10080), T1_10[:10080], c='red', label='T1_10')
ax1.plot(np.arange(10080), T1_100[:10080], c='orange', label='T1_100')
ax1.plot(np.arange(10080), T1_1000[:10080], c='purple', label='T1_1000')
ax1.plot(np.arange(10080), T5[:10080], c='green', label='T5')
plt.legend(loc='upper left')
plt.xlabel('Time (minutes)')
plt.ylabel('Power (W)')
plt.show()

############ GET DAILY USAGE #####################

# Get an array of energies in kWh for each minute
T1_energy = (T1/1000)*(1/60)
T2_energy = (T2/1000)*(1/60)
T3_energy = (T3/1000)*(1/60)
T4_energy = (T4/1000)*(1/60)
T5_energy = (T5/1000)*(1/60)

# Define lists to hold daily energies
T1_daily_energies = []
T2_daily_energies = []
T3_daily_energies = []
T4_daily_energies = []
T5_daily_energies = []

# For each day
for i in range(365):
    # Sum the energies over that day
    sum1 = sum(T1_energy[(i * 1440):(i * 1440 + 1440)])
    sum2 = sum(T2_energy[(i * 1440):(i * 1440 + 1440)])
    sum3 = sum(T3_energy[(i * 1440):(i * 1440 + 1440)])
    sum4 = sum(T4_energy[(i * 1440):(i * 1440 + 1440)])
    sum5 = sum(T5_energy[(i * 1440):(i * 1440 + 1440)])
    # Append the sum to the list
    T1_daily_energies.append(sum1)
    T2_daily_energies.append(sum2)
    T3_daily_energies.append(sum3)
    T4_daily_energies.append(sum4)
    T5_daily_energies.append(sum5)

# Save daily energies
np.savetxt("C:\\Users\\alyci\\Documents\\DPhil\\Working\\Demand\\T1_rand\\T1_daily_energies.csv",
           np.array(T1_daily_energies), delimiter=",")
np.savetxt("C:\\Users\\alyci\\Documents\\DPhil\\Working\\Demand\\T2_rand\\T2_daily_energies.csv",
           np.array(T2_daily_energies), delimiter=",")
np.savetxt("C:\\Users\\alyci\\Documents\\DPhil\\Working\\Demand\\T3_rand\\T3_daily_energies.csv",
           np.array(T3_daily_energies), delimiter=",")
np.savetxt("C:\\Users\\alyci\\Documents\\DPhil\\Working\\Demand\\T4_rand\\T4_daily_energies.csv",
           np.array(T4_daily_energies), delimiter=",")
np.savetxt("C:\\Users\\alyci\\Documents\\DPhil\\Working\\Demand\\T5_rand\\T5_daily_energies.csv",
           np.array(T5_daily_energies), delimiter=",")

