# Heatmaps for costs/feasible spaces at different MTF tiers for thesis
# Author: Alycia Leonard, Energy and Power Group, University of Oxford
# Contact: alycia.leonard@eng.ox.ac.uk

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

# Define file locations


working_path = 'C:\\Users\\alyci\\Documents\\DPhil\\Working\\HOMER\\Exp2_LCOE\\'

Combined = working_path + 'Iteration_3\\comb_affordability.csv'
Cost = working_path + 'Iteration_3\\Cost\\Exp2_LCOE_Cost_I3_SearchSpace_Results_Plot.csv'
T1_100 = working_path + 'Iteration_3\\T1_100\\Exp2_LCOE_T1_100_I3_SearchSpace3_Results_Plot.csv'
T2_100 = working_path + 'Iteration_3\\T2_100\\Exp2_LCOE_T2_100_I3_SearchSpace3_Results_Plot.csv'
T3_100 = working_path + 'Iteration_3\\T3_100\\Exp2_LCOE_T3_100_I3_SearchSpace3_Results_Plot.csv'

# # 90% capacity cap
# T2_100 = working_path + 'Iteration_3\\T2_100\\Exp2_LCOE_T2_100_I3_SearchSpace2_90cap_Results_Plot_Affordability2.csv'
# Combined = working_path + 'Iteration_3\\comb_90cap_affordability2.csv'

# Read files

df_T1 = pd.read_csv(T1_100, header=0)
df_T2 = pd.read_csv(T2_100, header=0)
df_T3 = pd.read_csv(T3_100, header=0)
df_comb = pd.read_csv(Combined, header=0)
df_cost = pd.read_csv(Cost, header=0)

# Make plots

fig, ax = plt.subplots()
sc = ax.scatter(df_T1.PV_kW, df_T1.Li_kWh, c=df_T1.LCOE, cmap="plasma", norm=matplotlib.colors.LogNorm(vmin=0.01, vmax=10))
cbar = fig.colorbar(sc, ax=ax, ticks=[0.01, 0.03, 0.1, 0.3, 1, 3, 10], format='%.2f')
cbar.set_label('LCOE ($)')
plt.yscale('log')
plt.xscale('log')
ax.set_xlim([1,250])
ax.set_ylim([1,350])
plt.title('T1')  # ($0.27/kWh affordability cap)')
plt.xlabel('PV (kW)')
plt.ylabel('Li-ion Battery (kWh)')

fig, ax = plt.subplots()
sc = ax.scatter(df_T2.PV_kW, df_T2.Li_kWh, c=df_T2.LCOE, cmap="plasma") #, norm=matplotlib.colors.LogNorm(vmin=0.01, vmax=10))
cbar = fig.colorbar(sc, ax=ax) #, ticks=[0.01, 0.03, 0.1, 0.3, 1, 3, 10], format='%.2f')
cbar.set_label('LCOE ($)')
#plt.yscale('log')
#plt.xscale('log')
#x.set_xlim([1,250])
#ax.set_ylim([1,350])
plt.title('T2, 10% Capacity Shortage, $0.27/kWh affordability cap)')
plt.xlabel('PV (kW)')
plt.ylabel('Li-ion Battery (kWh)')

fig, ax = plt.subplots()
sc = ax.scatter(df_T3.PV_kW, df_T3.Li_kWh, c=df_T3.LCOE, cmap="plasma", norm=matplotlib.colors.LogNorm(vmin=0.01, vmax=10))
cbar = fig.colorbar(sc, ax=ax, ticks=[0.01, 0.03, 0.1, 0.3, 1, 3, 10], format='%.2f')
cbar.set_label('LCOE ($)')
plt.yscale('log')
plt.xscale('log')
ax.set_xlim([1,250])
ax.set_ylim([1,350])
plt.title('T3') # ($0.27/kWh affordability cap)')
plt.xlabel('PV (kW)')
plt.ylabel('Li-ion Battery (kWh)')

fig, ax = plt.subplots()
sc = ax.scatter(df_comb.PV_kW, df_comb.Li_kWh, c=df_comb.LCOE, cmap="plasma", vmin=0.10, vmax=0.27)
cbar = fig.colorbar(sc, ax=ax)
cbar.set_label('LCOE ($)')
#plt.clim(0.10,0.27)
plt.title('T1-T3 feasible spaces') # \n 10% capacity shortage, $0.27/kWh affordability cap')
plt.xlabel('PV (kW)')
plt.ylabel('Li-ion Battery (kWh)')
# ax.set_yscale('log')
# ax.set_xscale('log')

fig, ax = plt.subplots()
sc = ax.scatter(df_cost.PV_kW, df_cost.Li_kWh, c=df_cost.NPC_per_hh_per_year, cmap="plasma")
cbar = fig.colorbar(sc, ax=ax)
cbar.set_label('NPC ($)')
plt.title('System cost per household per year over PV & Li-Ion sizes \n(hh=100, lifetime=25yrs)')
plt.xlabel('PV (kW)')
plt.ylabel('Li-ion Battery (kWh)')

plt.show()


