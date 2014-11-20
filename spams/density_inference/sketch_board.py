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

if __name__ == "__main__":
    metadata, connection = setup_database()
    places_location = get_table("places_location", metadata)
    min_lat, max_lat, min_long, max_long = connection.execute(select([func.min(places_location.c.latitude), func.max(places_location.c.latitude), func.min(places_location.c.longitude), func.max(places_location.c.longitude)])).fetchall()[0]
    lat_long_query = select([places_location.c.latitude, places_location.c.longitude])
    results = connection.execute(lat_long_query).fetchall()
    xy = np.asarray([(float(r[0]), float(r[1])) for r in results])
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
        print len(indices)
        local = xy[indices]
        X = np.asarray([x[1] for x in local])
        Y = np.asarray([x[0] for x in local])
        m = Basemap(projection='cyl', llcrnrlat=float(min_lat) - 1, urcrnrlat=float(max_lat) + 1, llcrnrlon=float(min_long) - 1, urcrnrlon=float(max_long) + 1, resolution='c')
        x, y = m(X,Y)
        m.scatter(x,y,3,marker='o',color='k')
        m.drawcountries()
        plt.show()
