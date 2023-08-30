# Plot HRSL population maps
# Author: Alycia Leonard, Energy and Power Group, University of Oxford
# Contact: alycia.leonard@eng.ox.ac.uk

import geopandas as gp
import geoplot
import matplotlib.pyplot as plt

data = gp.read_file('population_ken_2018-10-01.csv')

world = gp.read_file(
    gp.datasets.get_path('naturalearth_lowres'))

ax = geoplot.kdeplot(data, # clip=boroughs.geometry,
    shade=True, cmap='Reds',
    projection=geoplot.crs.AlbersEqualArea())

geoplot.polyplot(world, ax=ax, zorder=1)

plt.show()