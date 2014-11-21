import sys
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
        colors.append(cm(1.*i/NUM_COLORS))
    return colors



if __name__ == "__main__":
    metadata, connection = setup_database()
    places_location = get_table("places_location", metadata)
    lat_long_query = select([places_location.c.latitude, places_location.c.longitude, places_location.c.place_label, places_location.c.userid, places_location.c.placeid])
    min_lat, max_lat, min_long, max_long = connection.execute(select([func.min(places_location.c.latitude), func.max(places_location.c.latitude), func.min(places_location.c.longitude), func.max(places_location.c.longitude)])).fetchall()[0]
    results = connection.execute(lat_long_query).fetchall()
    xy = np.asarray([(float(r[0]), float(r[1])) for r in results])
    places = [(r[3], r[4]) for r in results]
    semantic_labels = [r[2] for r in results]
    c_map = colors_map()
    label_color_dict = {}
    for i, label in enumerate(list(set(semantic_labels))):
        label_color_dict[label] = c_map[i]
    db = DBSCAN(eps =0.05, metric="haversine", algorithm="ball_tree", min_samples=2).fit(xy)
    labels = db.labels_
    # Number of clusters in labels, ignoring noise if present.
        #sys.exit(0)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    # print n_clusters
    indices_with_labels = enumerate(labels)
    
    def keyfunc(x):
        return x[1]

    indices_with_labels = sorted(indices_with_labels, key=keyfunc)
    index_label_dict = defaultdict(list)
    for key, group in groupby(indices_with_labels, keyfunc):
        index_label_dict[key].extend(group)
    max_label_indices = []
    for key in index_label_dict:
        if len(index_label_dict[key]) > len(max_label_indices):
            max_label_indices = index_label_dict[key]
   
    for label in set(labels):
        indices = [x[0] for x in enumerate(labels) if x[1] == label]
        semantic_labels_group = [semantic_labels[i] for i in indices]

        points = xy[indices]    
        points_with_sl = zip(semantic_labels_group, points)

        labels_group = sorted(semantic_labels_group, key=lambda x: x[0])
        labels_places_dict = defaultdict(list)
        for key, group in groupby(points_with_sl, lambda x:x[0]):
            labels_places_dict[key].extend((list(group)))
        
        m = Basemap(projection='cyl', llcrnrlat=float(min_lat) - 1, urcrnrlat=float(max_lat) + 1, llcrnrlon=float(min_long) - 1, urcrnrlon=float(max_long) + 1, resolution='c')
        for key in labels_places_dict:
            local = [p[1] for p in labels_places_dict[key]]
            #print local
            X = np.asarray([x[1] for x in local])
            Y = np.asarray([x[0] for x in local])
            x, y = m(X,Y)
            m.scatter(x,y,3,marker='o',color=label_color_dict[key])
            m.drawcountries()
            #plt.title("Count of points " + str(count))
    handles = []
    for key in label_color_dict:
        handles.append(mpatches.Patch(color = label_color_dict[key], label = key))

    plt.legend(handles = handles)    
    plt.show()

        # print sorted(labels_dict.items(), key=lambda x: x[1], reverse = True)     
        # plot_on_map(xy[indices])
