# Analysis for May 5th 2020 Zooniverse report
# Author: Alycia Leonard, Energy and Power Group, University of Oxford
# Contact: alycia.leonard@eng.ox.ac.uk

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import MO, TU, WE, TH, FR, SA, SU
import json
import numpy as np
import seaborn as sb


# Helper function to get UTC offset
def get_utc_offset(metadata_str):
    return float(json.loads(metadata_str).get('utc_offset'))


def get_T0_answer(annotations_str):
    return json.loads(annotations_str)[0].get("value")


def get_T1_box_count(annotations_str):
    if json.loads(annotations_str)[0].get("value") == "Yes":
        return len(json.loads(annotations_str)[1].get("value"))
    else:
        return "None"


def extract_T1_labels_to_list(annotations_str, new_list):
    if json.loads(annotations_str)[0].get("value") == "Yes":
        for i in json.loads(annotations_str)[1].get("value"):
            new_list.append(i)


def get_detail_1(details_list):
    return details_list[0].get("value")


def get_detail_2(details_list):
    return details_list[1].get("value")


# Settings
data = "C:\\Users\\alyci\\Documents\\DPhil\\GeoDESA\\Zooniverse\\DataExports\\DataExport_20200617\\power-to-the-people-classifications.csv"
week = mdates.WeekdayLocator(byweekday=SU)
auto = mdates.AutoDateLocator()

# Read classification data into pandas
pttp = pd.read_csv(data)

# Turn annotations, metadata, and subject_data columns into dicts
#pttp["annotations_dict"] = pttp["annotations"].map(json.loads)
#pttp["metadata_dict"] = pttp["metadata"].map(json.loads)
#pttp["subject_data_dict"] = pttp["subject_data"].map(json.loads)

# SCOT'S BASE WORK WITH MY MODIFICATIONS FROM HERE DOWN

# Preprocess data, add necessary columns, etc
pttp["classification"] = pttp.index.values
pttp["created_at"] = pd.to_datetime(pttp["created_at"])  # str to local timestamp, tz aware
pttp["utc_offset"] = pttp["metadata"].apply(get_utc_offset)/-3600  # hrs offset from UTC, -ive is west
pttp["local_created_at"] = pttp["created_at"].apply(lambda x: x.tz_localize(None))  # local timestamp, tz naive
pttp["local_hour"] = pttp["local_created_at"].apply(lambda x: x.hour)  # local hour
pttp["day"] = pttp["local_created_at"].apply(lambda x: x.dayofweek)  # day of week. 0 = Mon
pttp["day_name"] = pttp["local_created_at"].apply(lambda x: x.day_name())  # day of week. 0 = Mon
pttp["created_at_utc"] = pttp["created_at"].apply(lambda x: x.tz_convert('UTC'))  # UTC timestamp
pttp["created_at_utc"] = pttp["created_at_utc"].apply(lambda x: x.tz_localize(None))  # remove tz info
pttp["homes_detected"] = pttp["annotations"].apply(get_T0_answer)
pttp["number_homes_detected"] = pttp["annotations"].apply(get_T1_box_count)
pttp.set_index("created_at_utc", inplace=True, drop=False)  # set index to created_at_utc

#pd.set_option('display.max_columns', None)
#print(pttp["number_homes_detected"])

# Plot classifications since launch
mask = (pttp['created_at_utc'] > pd.datetime(2020, 3, 1)) #& (pttp['created_at_utc'] < pd.datetime(2020, 5, 11)) # Uncomment this for 8 weeks only for heatmap - account for even # of each day of week
pttp_live = pttp.loc[mask]
baseline = pttp["classification"][pttp_live.index.values[0]]
pttp_live.loc[:, "classification"] = pttp.loc[mask, "classification"].values - baseline
ax = pttp_live.plot(y="classification", legend=False, color='orange')
ax.set_xlabel('Date (UTC)')
ax.set_ylabel(ylabel='Classifications')
ax.xaxis.set_major_locator(week)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%d"))
plt.grid(axis='y', zorder=0)
plt.grid(axis='x', zorder=0)
plt.ylim(0, 400000)
plt.xticks(rotation='45')
plt.yticks(np.arange(0, 400000, 20000))
plt.tight_layout()
plt.show()

# Get a list of all the boxes drawn since launch
list_of_boxes = []
for index, row in pttp_live.iterrows():
    extract_T1_labels_to_list(row["annotations"], list_of_boxes)

# Turn list of boxes into pd dataframe
boxes = pd.DataFrame(list_of_boxes)
boxes["detail_1"] = boxes["details"].apply(get_detail_1)
boxes["detail_2"] = boxes["details"].apply(get_detail_2)
boxes.to_csv('C:\\Users\\alyci\\Documents\\DPhil\\GeoDESA\\Zooniverse\\DataExports\\DataExport_20200617\\boxes_since_launch.csv')

# Group by whether homes were seen
pttp_live_homespresent = pttp_live.groupby(by="homes_detected").count()[["classification_id"]]
pttp_live_homespresent.columns = ["classifications_where_homes_detected"]

# Group by number of homes in classification and save to CSV
pttp_live_homescount = pttp_live.groupby(by="number_homes_detected").count()[["classification_id"]]
pttp_live_homescount.columns = ["classifications_where_number_homes_detected"]
pd.options.display.max_rows = None
#pttp_live_homescount.to_csv("C:\\Users\\alyci\\Documents\\DPhil\\GeoDESA\\Zooniverse\\DataExports\\DataExport_20200514\\Classifications_Homes_Count.csv")

# Group by subject id (number of times each subject ID is classified) and save to CSV
pttp_live_subjectid = pttp_live.groupby(by="subject_ids").count()[["classification_id"]]
pttp_live_subjectid.columns = ["classification_count"]
#pttp_live_subjectid.to_csv("C:\\Users\\alyci\\Documents\\DPhil\\GeoDESA\\Zooniverse\\DataExports\\DataExport_20200514\\Classifications_Per_SubjectID.csv")

# Group by hour
pttp_live_hour = pttp_live.groupby(by="local_hour").count()[["classification_id"]]
pttp_live_hour.columns = ["hour_count"]

# Group by day
pttp_live_day = pttp_live.groupby(by="day").count()[["classification_id"]]
pttp_live_day.columns = ["day_count"]
pttp_live_day["day_name"] = ["Mon", "Tues", "Weds", "Thurs", "Fri", "Sat", "Sun"]

# Group by tz
pttp_live_tz = pttp_live.groupby(by="utc_offset").count()[["classification_id"]]
pttp_live_tz.columns = ["utc_offset_count"]

# Group by hour of day and day of week
pttp_live_heat = pttp_live.groupby(["day", "local_hour"]).count()[["classification_id"]]
pttp_live_heat.columns = ["hourofday_count"]
# Make into numpy array
heat_array = np.arange(168).reshape(24,7)
for index, row in pttp_live_heat.iterrows():
    heat_array[index[1]][index[0]] = row['hourofday_count']
# Make heat map of hour of day and day of week
heat_map = sb.heatmap(heat_array)

# Plot classifications by hour of day
hour = pttp_live_hour.plot(y="hour_count", kind="bar", legend=False, color="turquoise", zorder=3)
hour.set_xlabel('Hour of Day')
hour.set_ylabel('Classifications')
plt.grid(axis='y', zorder=0)
plt.tight_layout()
plt.show()

# Plot classifications by day of the week
day = pttp_live_day.plot(x="day_name", y="day_count", kind="bar", legend=False, color="purple", zorder=3)
day.set_xlabel('Day of Week')
day.set_ylabel('Classifications')
plt.grid(axis='y', zorder=0)
plt.tight_layout()
plt.show()

# Plot classifications by time zone
tz = pttp_live_tz.plot(y="utc_offset_count", kind="bar", legend=False, color="green", zorder=3)
tz.set_xlabel('UTC Offset')
tz.set_ylabel('Classifications')
plt.grid(axis='y', zorder=0)
plt.tight_layout()
plt.show()

