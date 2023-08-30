# Plots annual expenditures in Sierra Leone
# Author: Alycia Leonard, Energy and Power Group, University of Oxford
# Contact: alycia.leonard@eng.ox.ac.uk

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)

# Define filepaths
map_fp = "C:\\Users\\alyci\\PycharmProjects\\GeoDESA\\plotting\\sle_admbnda_adm1_1m_gov_ocha\\sle_admbnda_adm1_1m_gov_ocha_20161017.geojson"

# Read map geojson
map_df = gpd.read_file(map_fp)

# Set the range for the choropleth
vmin, vmax = 10000000, 40000000

# Create figure and axes for Matplotlib
fig, ax = plt.subplots(1, figsize=(8, 8))

# Create map
map_df.plot(column="annual_exp", cmap='Blues', linewidth=0.8, ax=ax, edgecolor='0.8') #, missing_kwds={"color": "lightgrey"})
# Turn off the axis
ax.axis('off')
# Create an annotation for the data source
ax.annotate('Source: 2018 Sierra Leone Integrated Household Survey', xy=(0.1, 0.15), xycoords='figure fraction', horizontalalignment='left', verticalalignment='top', fontsize=8, color='#555555')
# Create colorbar as a legend
sm = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(vmin=vmin, vmax=vmax))
# Empty array for the data range
sm._A = []
# Add the colorbar to the figure
cbar = fig.colorbar(sm, fraction=0.033, pad=0.04)

plt.show()

