# General LCOE exploration script based on tiers
# Author: Alycia Leonard, Energy and Power Group, University of Oxford
# Contact: alycia.leonard@eng.ox.ac.uk

# Select community parameters: X households in Y area
# For a range of battery and PV sizes:
# Calculate levelized cost of electricity at each, plot heatmap
# Determine tier of service available at each point
# Note: All costs in USD ($)

# From MTF report: Capacity is measured in watts for grids, mini-grids, and fossil-fuel-based generators, and in watt-hours
# for rechargeable batteries, solar lanterns, and solar home systems.

import numpy as np
import pandas as pd

# Set simulation parameters
pv_out = 1466  # PV OUT from Global Solar Atlas (kWh/kWp)
pv_cap = 210  # PV capital cost ($/kW) from IRENA "Renewable Power Generation Costs in 2019" - text, Fig 3.2 (given in $/W in text)
pv_om = 9.50  # PV O&M cost ($/kW per year)
bat_cap = 140  # battery capital cost ($/kWh)
bat_om = 10  # battery O&M cost ($/kWh per year)
bat_dod = 0.8  # battery depth of discharge (%)
gen_v = 12  # voltage for PV and battery (V)
dist_v = 230  # distribution voltage (V)
sys_discount = 0.08  # system discount (%)
sys_om_growth = 0.02  # annual growth in O&M cost
sys_lifetime = 25  # assumed system lifetime in years
sys_hh = 30  # number of households connected to system

# From LCOE spreadsheet: For 30 households, T5, 61.25 kW of PV, 922.5 kWh of battery
# Use input vectors from 0 -> 100 kW PV (steps of 0.1 kW), 0 -> 1000 kWh battery (steps of 1)

# Set PV and battery size input vectors (round to one decimal)
pv_sizes = np.around(np.linspace(0, 100, 1001), 1)
bat_sizes = np.around(np.linspace(0, 1000, 1001), 1)

# Create dataframe to hold results
results = pd.DataFrame(index=pv_sizes, columns=bat_sizes)

# Cycle through all design points
for i in range(1001):
    pv_size = pv_sizes[i]
    for j in range(1001):
        bat_size = bat_sizes[j]

        # Create dataframe for LCOE calculation
        df = pd.DataFrame(index=np.arange(sys_lifetime+1), columns=['capital', 'om', 'energy', 'discount_factor',
                                                            'NPV_costs', 'NPV_energy'])

        # Assign start-up capital costs and zero the rest of the column
        df['capital'] = 0
        df.at[0, 'capital'] = (pv_size * pv_cap) + (bat_size * bat_cap)

        # Assign O&M rates in each year (set year 0 (start-up costs row) to 0 thereafter)
        df['om'] = ((pv_size * pv_om) + (bat_size * bat_om)) * (1 + sys_om_growth) ** (df.index - 1)
        df.at[0, 'om'] = 0

        # Assign energy output in each year based on PV generation (i.e. kWh/kWp per year * kWp rating)
        # Battery just moves energy around anyway. Assume all energy is used, unsure if possible given ratings.
        df['energy'] = pv_size * pv_out

        # Assign discount rates in each year
        df['discount_factor'] = (1/(1+sys_discount)) ** df.index

        # Calculate NPV of costs each year
        df['NPV_costs'] = (df['capital'] + df['om']) * df['discount_factor']

        # Calculate NPV of energy production each year
        df['NPV_energy'] = df['energy'] * df['discount_factor']

        # Calculate LCOE
        LCOE = df['NPV_costs'].sum() / df['NPV_energy'].sum()

        # Print LCOE
        print('LCOE: $' + str(round(LCOE, 2)) + '/kWh (' + str(pv_size) + ' kW PV, ' + str(bat_size) + ' kWh batteries)')

        # Save in results dataframe
        results.at[pv_sizes[i], bat_sizes[j]] = LCOE

# Save results as CSV
results.to_csv('C:\\Users\\alyci\\PycharmProjects\\GeoDESA\\mtflcoe_test.csv')
