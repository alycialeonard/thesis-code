# Geospatial visualisation for SONG demographic data
# Author: Alycia Leonard, Energy and Power Group, University of Oxford
# Contact: alycia.leonard@eng.ox.ac.uk
# Date: 2020-07-15

import pandas as pd
import geopandas as gp
import plotly.graph_objects as go
import plotly

data_path = 'C:\\Users\\alyci\\Documents\\DPhil\\Paper Drafts\\Location+Appearance+Load+Demo\\households_EPSG4326_withdata_QGIS_private.csv'
hub_path = 'C:\\Users\\alyci\\Documents\\DPhil\\Paper Drafts\\Location+Appearance+Load+Demo\\solar_hubs.csv'
interactive_graph_save = 'C:\\Users\\alyci\\Documents\\DPhil\\Paper Drafts\\Location+Appearance+Load+Demo\\interactive_graph.html'

print('Loading SONG survey home location + demographic data...')
gdf = gp.read_file(data_path)
print('Loading hub locations...')
hubs = gp.read_file(hub_path)

# Making interactive HTML cluster visualisation
print("Creating interactive visualisation...")

fig = go.Figure()

fig.add_trace(go.Scattermapbox(
        lon=gdf["X"],
        lat=gdf["Y"],
        mode='markers',
        marker=dict(
            size=pd.to_numeric(gdf["people"]),
            sizemode="diameter",
            sizemin=5,
            sizeref=0.100, # - use this if I decide to set sizemode back to diameter
            color=pd.to_numeric(gdf["distance"]),
            colorscale="YlOrRd",
            showscale=True,
            opacity=0.7,
            colorbar=dict(
                title=dict(
                    text="Self-reported distance to hub"
                )
            )
        ),
        text=gdf['people'].astype(str) + ' people in the household<br>Battery charged ' + gdf['frequency'].astype(str) + " times per week<br>Grid connection status: " + gdf['grid-connected'].astype(str),
        hoverinfo='text'
    ))

fig.add_trace(go.Scattermapbox(
        lon=hubs["X"],
        lat=hubs["Y"],
        mode='markers',
        marker=dict(
            size=20,
            color="black",
            opacity=0.7,
        ),
        text=hubs['community'].astype(str),
        hoverinfo='text'
    ))

fig.update_layout(
    title='<b>Solar Nanogrid Demographic Surveys</b><br>Diameter proportional to # of people in the household',
    autosize=True,
    hovermode='closest',
    showlegend=False,
    mapbox=go.layout.Mapbox(
        style='open-street-map',
        center=dict(
            lon=pd.to_numeric(gdf["X"][1]),
            lat=pd.to_numeric(gdf["Y"][1])
        ),
        zoom=10
    ),
)

plotly.offline.plot(fig, filename=interactive_graph_save)

print("Plot closed!")

