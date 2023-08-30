# Stochastic load profile construction
# Method adapted from https://doi.org/10.1007/s12053-018-9725-6
# NOTE: Unfinished
# Author: Alycia Leonard, Energy and Power Group, University of Oxford
# Contact: alycia.leonard@eng.ox.ac.uk

import pandas as pd
import numpy as np
import random


# Create window vectors based on input data (1 = functional window, 0 = not in defined window)
def make_windows(df_row):
    windows = np.zeros([1440])
    w1_start = (df_row['W1_start'] * 60) -1
    #print(w1_start)
    w1_end = w1_start + df_row['W1_length'] * 60
    if w1_start == -1: # Case where the start is at midnight
        w1_start = 0
    #print(w1_end)
    windows[w1_start:w1_end + 1] = 1
    w2_start = (df_row['W2_start'] * 60) - 1
    #print(w2_start)
    w2_end = w2_start + df_row['W2_length'] * 60
    #print(w2_end)
    windows[w2_start:w2_end + 1] = 1
    return windows


# Load inputs, convert all columns to numeric, and make a copy with load type as index
input_df = pd.read_csv("load_profile_construction_inputs.csv")
input_df = input_df.apply(pd.to_numeric, errors='ignore')
input_df_2 = input_df.set_index("j", drop=False)

# Set coincidence factor (0.2 to 1)
cf = 0.2

# Determine Wpeak based on input window times as overlaps
# write code to compute Wpeak (currently imported as dataframe from excel calculations)
wpeak_df = pd.read_csv("wpeak.csv")
wpeak_df = wpeak_df.set_index("tier", drop=False)

# add preceding for loops: For each tier, For each day of the year

# Calculate the sum of each column in the input df
column_totals = input_df.sum(axis=0)

# Create empty array to hold results for each load in T1 (one row per individual load)
results_array = np.empty([column_totals['q_T1'], 1440])

# Create list of dictionaries to hold parameters needed to calculate results for each load
led_params = [{} for _ in range(input_df_2.loc['led_lighting', 'q_T1'])]
mobile_params = [{} for _ in range(input_df_2.loc['mobile_phone_charging', 'q_T1'])]
params = [led_params, mobile_params]

# Make window vectors (1440 long, 0 = not in functional window, 1 = in functional window)
led_windows = make_windows(input_df_2.loc['led_lighting'])
#print(input_df_2.loc['mobile_phone_charging'])
mobile_windows = make_windows(input_df_2.loc['mobile_phone_charging'])
#np.savetxt("foo.csv", mobile_windows, delimiter=",")
windows = [led_windows, mobile_windows]

# Make peak window vector (0 = not peak, 1 = peak)
peak_window = np.zeros([1440])
peak_start = (wpeak_df.loc['T1', 'start']*60)-1
peak_end = peak_start + (wpeak_df.loc['T1', 'length']*60)
peak_window[peak_start:peak_end+1] = 1

# Get vectors of indices where led and mobile are functional, and where peak exists
led_windows_indices = np.where(led_windows == 1)
mobile_windows_indices = np.where(mobile_windows == 1)
#print(mobile_windows_indices)
window_indices = [led_windows_indices, mobile_windows_indices]
peak_window_indices = np.where(peak_window == 1)
#print(peak_window_indices)

# For each load type (i.e. row in inputs)
for index, row in input_df.iterrows():
    # Proceed only for rows where there is T1 power usage
    if row["P_T1"] == 0:
        continue
    # For each load j of the load type
    for j in range(row['q_T1']):
        # Randomly generate number of instances for that load during the day between n_min and n_max
        n = random.randint(row['n_min'], row['n_max'])
        params[index][j]["n"] = n
        # Instantiate placeholder for total usage of this load
        usage = 0
        # Instantiate dynamically updated windows for this load
        dyn_window = windows[index]
        dyn_window_indices = window_indices[index]
        # For each load instance (i)
        for i in range(n):
            print(dyn_window_indices)
            # Randomly generate cycle time Tij between Tmin and Tmax
            T = random.randint(row['T_min'], row['T_max'])
            #label = "T_" + str(i)
            params[index][j]["T_" + str(i)] = T
            # Generate the start times of each instance
            if i == 0:
                # If first instance, force the start time t to be inside the peak window
                t = np.random.choice(peak_window_indices[0])
                #print("the first t is", t)
                #print(peak_window_indices[0])
            else:
                # Randomly generate a start time t for the instance from dynamically updating window
                if dyn_window_indices[0].size == 0:
                    break
                else:
                    t = np.random.choice(dyn_window_indices[0])
            #print("t is ", t)
            #print(window_indices[index][0])
            # Calculate the time at the end of the interval
            t_end = t + T
            # Check to see if t_end is still within available windows
            if t_end in dyn_window_indices[0]:
                pass
            else:
                pass
                # Set t_end = latest time in the window we're starting in
                #current = window_indices[index][0][0]
                #current = dyn_window_indices[0][0]
                #for element in dyn_window_indices[0]:
                    #if (abs(t_end - element) < abs(t_end - current)) and np.sign(t_end - element) == 1:
                        #current = element
                        #t_end = current
            # Get duration of time on
            # t_duration = t_end - t
            # Add duration to usage counter
            # usage = usage + t_duration
            # # Check to see if usage exceeds total allowed usage per load
            # if usage < (row['Tm_T1']*60):
            #     pass
            # else:
            #     # Set t_end so that usage = Tm_T1
            #     t_end = t_end - (usage - (row['Tm_T1']*60))
            #     usage = row['Tm_T1']*60
            # Save t and t_end parameters
            params[index][j]["t_start_" + str(i)] = t
            params[index][j]["t_end_" + str(i)] = t_end
            # Update window indices
            dyn_window[t:t_end+1] = 0
            dyn_window_indices = np.where(dyn_window == 1)

print(params)

# # Create usage window vectors for each load type
# led_windows = np.zeros([1440])
# led_w1_start_mins = (input_df_2.loc['led_lighting', 'W1_start']*60)-1
# led_w1_end_mins = led_w1_start_mins + (input_df_2.loc['led_lighting', 'W1_length']*60)
# led_windows[led_w1_start_mins:led_w1_end_mins+1] = 1
# led_w2_start_mins = (input_df_2.loc['led_lighting', 'W2_start']*60)-1
# led_w2_end_mins = led_w2_start_mins + (input_df_2.loc['led_lighting', 'W2_length']*60)
# led_windows[led_w2_start_mins:led_w2_end_mins+1] = 1
