# Investigate how many classifications are needed for image retirement
# Author: Alycia Leonard, Energy and Power Group, University of Oxford
# Contact: alycia.leonard@eng.ox.ac.uk

import pandas
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

classification_export = 'C:\\Users\\alyci\\Documents\\DPhil\\GeoDESA\\Zooniverse\\DataExports\\DataExport_20200303\\power-to-the-people-classifications.csv'

data = pandas.read_csv(classification_export)

data["annotations"]  = data["annotations"].map(json.loads)
data["metadata"]     = data["metadata"].map(json.loads)
data["subject_data"] = data["subject_data"].map(json.loads)

"""
Classifications export columns:

"annotations" column data looks like this: 
[
    {"task":"T0", "task_label":"","value":""},
    {"task":"T1","task_label":"","value": [
            {"x":0,"y":0,"tool":0,"frame":0,"width":0,"height":0,"details":[{"value":0},{"value":0}],"tool_label":""},
            {"x":0,"y":0,"tool":0,"frame":0,"width":0,"height":0,"details":[{"value":0},{"value":0}],"tool_label":""},
            {"x":0,"y":0,"tool":0,"frame":0,"width":0,"height":0,"details":[{"value":0},{"value":0}],"tool_label":""},
            {"x":0,"y":0,"tool":0,"frame":0,"width":0,"height":0,"details":[{"value":0},{"value":0}],"tool_label":""},
            {"x":0,"y":0,"tool":0,"frame":0,"width":0,"height":0,"details":[{"value":0},{"value":0}],"tool_label":""}
        ]
    }
]   

"metadata" column looks like this:
{
    "source":"",
    "session":"",
    "viewport":
        {"width":0, "height":0},
    "started_at":"timestamp",
    "user_agent":"Browser/OS",
    "utc_offset":"",
    "finished_at":"timestamp",
    "live_project":false,
    "interventions":
        {"opt_in":true,"messageShown":false},
    "user_language":"en",
    "user_group_ids":[],
    "subject_dimensions":
        [{"clientWidth":0, "clientHeight":0, "naturalWidth":0,"naturalHeight":0}, null, null],
    "subject_selection_state":
        {"retired":false, "selected_at":"timestamp", "already_seen":false, "selection_state":"", "finished_workflow":false, "user_has_finished_workflow":false},
    "workflow_translation_id":""
}

"subject_data" column looks like this: 
{
    "37481178":{
        "retired":
            {"id":0,
             "workflow_id":0,
             "classifications_count":0,
             "created_at":timestamp,
             "updated_at":timestamp,
             "retired_at":timestamp,
             "subject_id":0,
             "retirement_reason":string},
        "info":"",
        "ms_image":"filename.ext",
        "ps_image":"filename.ext",
        "pan_image":"filename.ext",
        "attribution":""
    }
}

{"41126918":{"retired":{"id":,"workflow_id":,"classifications_count":,"created_at":,"updated_at":,"retired_at":,"subject_id":,"retirement_reason":},"Info":"Near Bo, Southern Province, Sierra Leone","Attribution":"Imagery provided by Earth-i Ltd.","Panchromatic":"Bo_1_20180124_Pan_Rend_103_074.png","Pansharpened":"Bo_1_20180124_PS_GDAL_Rend_103_074.png","Multispectral":"Bo_1_20180124_MS_RGB_Rend_103_074.png"}}


"""

# Example: Print the x-coordinate of a given identified point.
# classification 1, task 1, value recorded, access dict of data, grab points, access dict of data, grab x
print(data["annotations"][69]) # "annotations" - index - task # - "value" (cycle through all items) -
print("x coordinate: ", data["annotations"][69][1]["value"][0]['x'])
print("y coordinate: ", data["annotations"][69][1]["value"][0]['y'])
print('width: ', data["annotations"][69][1]["value"][0]['width'])
print('height: ', data["annotations"][69][1]["value"][0]['height'])

# subject_ids for subjects which have 5 or more classifications done in the beta data export (2020-02-24)
# multi_class_ids = [38633745, 38633752, 38633844, 38633790, 38633823, 38633815, 38633827, 38633840, 38633889, 38633885,
#                   38633780, 38633819, 38633857, 38633852, 38633831, 38633866, 38633754, 38633746, 38633822, 38633877,
#                   38633893, 38633816, 38633755, 38633859, 38633766, 38633880, 38633788, 38633777, 38633856, 38633888,
#                   38633736, 38633806, 38633787, 38633891, 38633814, 38633895, 38633749]

# subject ids for subjects which have 3-5 clssifications in first 2 days of soft launch

multi_class_ids = [40487769, 40487351, 40488519, 40470353, 40470706, 40470615, 40470708, 40470483, 40470362, 40472686,
                   40471385, 40472464, 40472611, 40472540, 40486132, 40470423, 40470338, 40470441, 40169580, 40470370,
                   40486021, 40472283, 40472131, 40472088, 40471290, 40472648, 40472526, 40484612, 40484064, 40472406,
                   40487686, 40472126, 40483845, 40472481, 40472633, 40471601, 40487892, 40488409, 40483764, 40488003,
                   40472033, 40472171, 40485622, 40471268, 40488638, 40472678, 40488840, 40488077, 40472671, 40485539,
                   40485813, 40472362, 40483715, 40471932, 40486592, 40471708, 40472442, 40485908, 40485942, 40472111,
                   40486065, 40471419, 40471166, 40471388, 40471378, 40470570, 40472054, 40487726, 40169541, 40489379,
                   40471643, 40487386, 40471839, 40472099, 40484444, 40470484, 40470392, 40471533, 40472293, 40470707,
                   40470468, 40484428, 40471175, 40471341, 40472245, 40472684, 40483829, 40169598, 40470403, 40484817,
                   40472533, 40470429, 40470347, 40470517, 40471131, 40470322, 40470358, 40169635, 40470537, 40472105,
                   40486844, 40484361, 40486988, 40489231, 40472517, 40471460, 40471297, 40485438, 40489030, 40472104,
                   40470702, 40485774, 40484853, 40488266, 40471753, 40486320, 40471384, 40483807, 40472100, 40471590,
                   40485591, 40470596, 40470351, 40472504,  40471773]

# Get a subset of the data for each id where multiple classifications were made
for id in multi_class_ids:
    sub = data.loc[data['subject_ids'] == id]
    # Place to save task results
    T0 = []
    T1 = []
    fig, ax = plt.subplots(1)
    # Iterate over each row of the subset of the dataframe
    for index, row in sub.iterrows():
        # Iterate over each task completed
        for i in range(len(sub["annotations"][index])):
            #print(index, "Response to task ", i, ":", sub["annotations"][index][i]['value'])
            if i == 0:
                T0.append({index: sub["annotations"][index][i]['value']})
            if i == 1:
                x1 = sub["annotations"][index][i]['value'][0]['x']
                y1 = sub["annotations"][index][i]['value'][0]['y']
                width = sub["annotations"][index][i]['value'][0]['width']
                height = sub["annotations"][index][i]['value'][0]['height']
                T1.append({index: {"x1": x1, "y1": y1, "w": width, "h": height}})
                rect = Rectangle((x1, y1), width, height, linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
    print("Responses to T0 for subject ", id, ": ", len(T0))
    print("Responses to T1 for subject ", id, ": ", len(T1))
    title = str(id)
    ax.title.set_text("Classifications for Subject ID " + title)
    ax.scatter(x=0, y=0, s=20, alpha=0.01)
    ax.scatter(x=256, y=256, s=20, alpha=0.01)
    ax.autoscale()
    plt.savefig("C:\\Users\\alyci\\Documents\\DPhil\\GeoDESA\\Zooniverse\\DataExports\\DataExport_20200303\\RetirementSoftLaunchAnalysis\\" + title + ".png")
    #plt.show()

#print(T0)
#print(T1)


