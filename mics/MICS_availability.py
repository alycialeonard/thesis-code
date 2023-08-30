# Plots availability of of MICS data around the world
# Author: Alycia Leonard, Energy and Power Group, University of Oxford
# Contact: alycia.leonard@eng.ox.ac.uk

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)

# Define filepaths
map_fp = "C:\\Users\\alyci\\PycharmProjects\\GeoDESA\\plotting\\ne_50m_admin_0_countries\\ne_50m_admin_0_countries.shp"
mics_data_fp = "C:\\Users\\alyci\\PycharmProjects\\GeoDESA\\plotting\\MICS_appliance_data_available.csv"

# Read map shapefile and mics data csv
map_df = gpd.read_file(map_fp)
mics_df = pd.read_csv(mics_data_fp)

# Merge mics data with Africa gdf
merged_mics_df = map_df.set_index('ISO_A3').join(mics_df.set_index('Code'), how='left')

# Set the range for the choropleth
vmin, vmax = 0, 1

# Create figure and axes for Matplotlib
fig, ax = plt.subplots(1, figsize=(8, 8))

# Create map
merged_mics_df.plot(column='MICS', cmap='Blues', linewidth=0.8, ax=ax, edgecolor='0.8') #, missing_kwds={"color": "lightgrey"})
# Turn off the axis
ax.axis('off')
manager = plt.get_current_fig_manager()
manager.full_screen_toggle()
# ax.set_title('', fontdict={'fontsize': '12', 'fontweight': '3'}) # (% of people ages 15 and above)
# ax.annotate('', xy=(0.1, 0.15), xycoords='figure fraction', horizontalalignment='left', verticalalignment='top', fontsize=8, color='#555555')
# sm = plt.cm.ScalarMappable(cmap='Greens', norm=plt.Normalize(vmin=vmin, vmax=vmax))
# sm._A = []
# cbar = fig.colorbar(sm, fraction=0.033, pad=0.04)
plt.show()

