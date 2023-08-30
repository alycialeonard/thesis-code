# Plot classifications over time - code adapted from Scot
# Author: Alycia Leonard, Energy and Power Group, University of Oxford
# Contact: alycia.leonard@eng.ox.ac.uk


import pandas as pd
import pytz
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import MO, TU, WE, TH, FR, SA, SU
import json
import numpy as np
import datetime
import seaborn as sb

data = 'C:\\Users\\alyci\\OneDrive - Nexus365\\Data\\Zooniverse\\DataExports\\DataExport_20200922\\power-to-the-people-classifications.csv'
locator = mdates.AutoDateLocator() # mdates.WeekdayLocator(byweekday=SU)
formatter = mdates.AutoDateFormatter(locator)


# Helper function to get UTC offset
def get_utc_offset(metadata_str):
    return float(json.loads(metadata_str).get('utc_offset'))


# Read classification data into pandas
pttp = pd.read_csv(data)
pttp["classification"] = pttp.index.values

# not sure if created at, started at, finished at is UTC or local time?
pttp["created_at"] = pd.to_datetime(pttp["created_at"]) # str to local timestamp, tz aware
pttp["utc_offset"] = pttp["metadata"].apply(get_utc_offset)/-3600 # hrs offset from UTC, -ive is west
pttp["local_created_at"] = pttp["created_at"].apply(lambda x: x.tz_localize(None)) # local timestamp, tz naive
pttp["local_hour"] = pttp["local_created_at"].apply(lambda x: x.hour) # local hour
pttp["day"] = pttp["local_created_at"].apply(lambda x: x.dayofweek) # day of week. 0 = Mon
pttp["day_name"] = pttp["local_created_at"].apply(lambda x: x.day_name()) # day of week. 0 = Mon
pttp["created_at_utc"] = pttp["created_at"].apply(lambda x: x.tz_convert('UTC')) # UTC timestamp
pttp["created_at_utc"] = pttp["created_at_utc"].apply(lambda x: x.tz_localize(None)) # remove tz info
pttp.set_index("created_at_utc", inplace=True, drop=False) # set index to

# Plot classifications since launch
mask = (pttp['created_at_utc'] > pd.datetime(2020, 3, 1)) & (pttp['created_at_utc'] < pd.datetime(2020, 8, 29))
pttp_live = pttp.loc[mask]
baseline = pttp["classification"][pttp_live.index.values[0]]
pttp_live.loc[:, "classification"] = pttp.loc[mask, "classification"].values - baseline
ax = pttp_live.plot(y="classification", legend=False, color='orange')
ax.set_xlabel('Date')
ax.set_ylabel(ylabel='Classifications to date')
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter) #mdates.DateFormatter("%b-%d"))
plt.grid(axis='y', zorder=0)
plt.grid(axis='x', zorder=0)
plt.ylim(0, 520000)
plt.xlim(datetime.date(2020, 3, 1), datetime.date(2020, 9, 1))
#plt.xticks(rotation='90')
plt.yticks(np.arange(0, 520000, 100000))
plt.minorticks_on()
ax.tick_params(axis='x', which='minor', bottom=False)
ax.grid()
ax.grid(axis='y', which='minor', linestyle=':', color='grey') # linewidth='0.5',
ax.grid(which='major')
#plt.tight_layout()
plt.show()





# print(pttp.head())
#pttp["utc_offset"] = pttp["created_at"].apply(lambda x: x.utcoffset().total_seconds()/3600) # hrs offset from UTC, -ive towards US
# Print an example of the metadata
# metadata_example = pttp["metadata"][0]
# print(metadata_example)

# Plot all classifications over time
# pttp.plot(y="classification")
# plt.show()

# # Group by hour
# pttp_live_hour = pttp_live.groupby(by="local_hour").count()[["classification_id"]]
# pttp_live_hour.columns = ["hour_count"]
# print(pttp_live_hour)
#
# # Group by day
# pttp_live_day = pttp_live.groupby(by="day").count()[["classification_id"]]
# pttp_live_day.columns = ["day_count"]
# pttp_live_day["day_name"] = ["Mon", "Tues", "Weds", "Thurs", "Fri", "Sat", "Sun"]
# print(pttp_live_day)
#
# # Group by hour of day and day of week
# pttp_live_heat = pttp_live.groupby(["day", "local_hour"]).count()[["classification_id"]]
# pttp_live_heat.columns = ["hourofday_count"]
# #print(pttp_live_heat)
#
# # Make numpy array of this data
# heat_array = np.arange(168).reshape(24,7)
# #print(heat_array)
#
# # Put pttp_live_heat data into the array
# for index, row in pttp_live_heat.iterrows():
#     #print(index[0], index[1], row["hourofday_count"])
#     heat_array[index[1]][index[0]] = row['hourofday_count']
#
# #print(heat_array)
#
# # Make heat map of data
# heat_map = sb.heatmap(heat_array)
#
# # Group by tz
# pttp_live_tz = pttp_live.groupby(by="utc_offset").count()[["classification_id"]]
# pttp_live_tz.columns = ["utc_offset_count"]
#
# # Plot classifications by hour of day
# hour = pttp_live_hour.plot(y="hour_count", kind="bar", legend=False, color="turquoise", zorder=3)
# hour.set_xlabel('Hour of Day')
# hour.set_ylabel('Classifications')
# plt.grid(axis='y', zorder=0)
# plt.tight_layout()
# #plt.show()
#
# # Plot classifications by day of the week
# day = pttp_live_day.plot(x="day_name", y="day_count", kind="bar", legend=False, color="purple", zorder=3)
# day.set_xlabel('Day of Week')
# day.set_ylabel('Classifications')
# plt.grid(axis='y', zorder=0)
# plt.tight_layout()
# #plt.show()
#
# # Plot classifications by time zone
# tz = pttp_live_tz.plot(y="utc_offset_count", kind="bar", legend=False, color="green", zorder=3)
# tz.set_xlabel('UTC Offset')
# tz.set_ylabel('Classifications')
# plt.grid(axis='y', zorder=0)
# plt.tight_layout()
#
# plt.show()
