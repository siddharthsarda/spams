import os
os.environ['MPLCONFIGDIR'] = "/local/.config/matplotlib"

import sys
import csv
import numpy as np
from sklearn.cluster import DBSCAN
from sqlalchemy.sql import select
from sqlalchemy import func
from spams.db.utils import setup_database, get_table
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from spams.mappings import LABEL_PLACE_MAPPING
from itertools import groupby
from collections import defaultdict
import matplotlib.patches as mpatches

def plot_on_map(xy):
    pass    


def colors_map():
    NUM_COLORS = 10
    colors = []
    cm = plt.get_cmap('gist_rainbow')
    for i in range(NUM_COLORS):
        colors.append(cm(10.*i/NUM_COLORS))
    return colors



if __name__ == "__main__":
    metadata, connection = setup_database()
    places_location = get_table("places_location", metadata)
    restrict_to = []
    with open("biggest_cluster_places") as f:
        for line in f:
            line = line.strip()
            restrict_to.append(line)

    lat_long_query = select([places_location.c.latitude, places_location.c.longitude, places_location.c.place_label, places_location.c.id]).where(places_location.c.place_label == 'Work')
    queries = select([func.min(places_location.c.latitude), func.max(places_location.c.latitude), func.min(places_location.c.longitude), func.max(places_location.c.longitude)])
    min_lat, max_lat, min_long, max_long = connection.execute(queries).fetchall()[0]
    results = connection.execute(lat_long_query).fetchall()
    places_with_labels = [(r[2],(float(r[0]), float(r[1]))) for r in results]
    places = [r[3] for r in results]
    labels_places_dict = defaultdict(list)
    labels_places_count = defaultdict(int)
    places_with_labels = sorted(places_with_labels, key = lambda x:x[0])
    for key, group in groupby(places_with_labels, lambda x:x[0]):
        l = list(group)
        l = [a[1] for a in l]
        labels_places_dict[key].extend(l)
    #print labels_places_dict 
    c_map = colors_map()
    label_color_dict = {}
    for i, label in enumerate(labels_places_dict.keys()):
        label_color_dict[label] = c_map[i]

    import folium
    print folium.__file__
    
    map_osm = folium.Map(location=[46.5236, 6.53], zoom_start=15, width=1500, height=1000)
    count = 0
    for key in labels_places_dict:
        print label_color_dict[key]
        for val in labels_places_dict[key]:
            count += 1
            print count
            map_osm.simple_marker(val, popup_on=False, marker_color=label_color_dict[key])
    map_osm.create_map(path= "overall.html")

    # plot_on_map(xy[indices])
