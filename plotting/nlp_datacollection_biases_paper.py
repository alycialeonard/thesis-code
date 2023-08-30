# Plots literacy, urbanity, and internet access in Africa
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
lit_data_fp = "C:\\Users\\alyci\\PycharmProjects\\GeoDESA\\plotting\\Adult_Literacy_WB_Africa.csv"
rural_data_fp = "C:\\Users\\alyci\\PycharmProjects\\GeoDESA\\plotting\\Rural_Population_WB_Africa.csv"
urban_data_fp = "C:\\Users\\alyci\\PycharmProjects\\GeoDESA\\plotting\\Urban_Population_WB_Africa.csv"
internet_data_fp = "C:\\Users\\alyci\\PycharmProjects\\GeoDESA\\plotting\\Internet_Usage_WB_Africa.csv"

# Read map shapefile and literacy data csv
map_df = gpd.read_file(map_fp)
lit_df = pd.read_csv(lit_data_fp)
rural_df = pd.read_csv(rural_data_fp)
urban_df = pd.read_csv(urban_data_fp)
internet_df = pd.read_csv(internet_data_fp)

# Get only African countries from map
is_africa = map_df['CONTINENT'] == 'Africa'
africa_df = map_df[is_africa].copy()

# Merge literacy data with Africa gdf
merged_lit_df = africa_df.set_index('ISO_A3').join(lit_df.set_index('Code'), how='left')

# Set the range for the choropleth
vmin, vmax = 0, 100

# Create figure and axes for Matplotlib
fig, ax = plt.subplots(1, figsize=(8, 8))

# Create map
merged_lit_df.plot(column='Rate', cmap='Blues', linewidth=0.8, ax=ax, edgecolor='0.8', missing_kwds={"color": "lightgrey"})
# Turn off the axis
ax.axis('off')
# Add a title
#ax.set_title('Adult Literacy Rate (% of people ages 15 and above)', fontdict={'fontsize': '12', 'fontweight': '3'}) # (% of people ages 15 and above)
# Create an annotation for the data source
#ax.annotate('Source: UNESCO Institute for Statistics', xy=(0.1, 0.15), xycoords='figure fraction', horizontalalignment='left', verticalalignment='top', fontsize=8, color='#555555')
# Create colorbar as a legend
sm = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(vmin=vmin, vmax=vmax))
# Empty array for the data range
sm._A = []
# Add the colorbar to the figure
cbar = fig.colorbar(sm, fraction=0.033, pad=0.04)

plt.show()


# Merge rurality data with Africa gdf
merged_rural_df = africa_df.set_index('ISO_A3').join(rural_df.set_index('Code'), how='left')

# Create figure and axes for Matplotlib
fig, ax = plt.subplots(1, figsize=(8, 8))
# Create map
merged_rural_df.plot(column='2019', cmap='Reds', linewidth=0.8, ax=ax, edgecolor='0.8', missing_kwds={"color": "lightgrey"})
# Turn off the axis
ax.axis('off')
# Add a title
#ax.set_title('Rural population (% of total population)', fontdict={'fontsize': '12', 'fontweight': '3'})
# Create an annotation for the data source
# ax.annotate('Source: World Bank', xy=(0.15, 0.15), xycoords='figure fraction', horizontalalignment='left', verticalalignment='top', fontsize=8, color='#555555')
# Create colorbar as a legend
sm = plt.cm.ScalarMappable(cmap='Reds', norm=plt.Normalize(vmin=vmin, vmax=vmax))
# Empty array for the data range
sm._A = []
# Add the colorbar to the figure
cbar = fig.colorbar(sm, fraction=0.033, pad=0.04)

plt.show()

# Merge urban pop data with Africa gdf
merged_urban_df = africa_df.set_index('ISO_A3').join(urban_df.set_index('Code'), how='left')

# Create figure and axes for Matplotlib
fig, ax = plt.subplots(1, figsize=(8, 8))
# Create map
merged_urban_df.plot(column='2019', cmap='Reds', linewidth=0.8, ax=ax, edgecolor='0.8', missing_kwds={"color": "lightgrey"})
# Turn off the axis
ax.axis('off')
# Add a title
#ax.set_title('Urban population (% of total population)', fontdict={'fontsize': '12', 'fontweight': '3'})
# Create an annotation for the data source
#ax.annotate('Source: World Bank', xy=(0.2, 0.15), xycoords='figure fraction', horizontalalignment='left', verticalalignment='top', fontsize=8, color='#555555')
# Create colorbar as a legend
sm = plt.cm.ScalarMappable(cmap='Reds', norm=plt.Normalize(vmin=vmin, vmax=vmax))
# Empty array for the data range
sm._A = []
# Add the colorbar to the figure
cbar = fig.colorbar(sm, fraction=0.033, pad=0.04)

plt.show()

# Merge internet data with Africa gdf
merged_internet_df = africa_df.set_index('ISO_A3').join(internet_df.set_index('Code'), how='left')

# Create figure and axes for Matplotlib
fig, ax = plt.subplots(1, figsize=(8, 8))
# Create map
merged_internet_df.plot(column='Rate', cmap='Purples', linewidth=0.8, ax=ax, edgecolor='0.8', missing_kwds={"color": "lightgrey"})
# Turn off the axis
ax.axis('off')
# Add a title
#ax.set_title('Individuals using the Internet (% of population)', fontdict={'fontsize': '12', 'fontweight': '3'})
# Create an annotation for the data source
# ax.annotate('Source: International Telecommunication Union (ITU)', xy=(0.2, 0.15), xycoords='figure fraction', horizontalalignment='left', verticalalignment='top', fontsize=8, color='#555555')
# Create colorbar as a legend
sm = plt.cm.ScalarMappable(cmap='Purples', norm=plt.Normalize(vmin=vmin, vmax=vmax))
# Empty array for the data range
sm._A = []
# Add the colorbar to the figure
cbar = fig.colorbar(sm, fraction=0.033, pad=0.04)

plt.show()
