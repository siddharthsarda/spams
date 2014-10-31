import sys
import numpy as np
import math
from sklearn.neighbors import KernelDensity
from sqlalchemy.sql import select
from sqlalchemy import func
from spams.db.utils import setup_database, get_table
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from spams.mappings import LABEL_PLACE_MAPPING
from sklearn.grid_search import GridSearchCV


if __name__ == "__main__":
    metadata, connection = setup_database()
    places_location = get_table("places_location", metadata)
    min_lat, max_lat, min_long, max_long = connection.execute(select([func.min(places_location.c.latitude), func.max(places_location.c.latitude), func.min(places_location.c.longitude), func.max(places_location.c.longitude)])).fetchall()[0]
    estimators = {}
    test_set = {}
    training_set = []
    for label in xrange(1, 11):
        lat_long_query = select([places_location.c.latitude, places_location.c.longitude]).where(places_location.c.place_label_int==label)
        results = connection.execute(lat_long_query).fetchall()
        result_len = len(results)
        test_len = math.ceil(result_len * 0.1)
        latitudes = [float(r[0]) for r in results]
        longitudes = [float(r[1]) for r in results]
        xy = zip(latitudes, longitudes)
        test_indices = np.random.choice(result_len, test_len)
        training_set =[]
        label = LABEL_PLACE_MAPPING[label]
        test_set[label] = []
        training_set = []
        for index, val in enumerate(xy):
            if index in test_indices:
                test_set[label].append(val)
            else:
                training_set.append(val)

        xy = np.array(training_set)
        xy *= np.pi /180.
        params = {'bandwidth': np.logspace(-3, 3, 20)}
        grid = GridSearchCV(KernelDensity(), params)
        grid.fit(xy)
        estimators[label] = grid.best_estimator_
        print grid.best_estimator_.bandwidth
    accurate = 0.0
    counter = 0.0
    for label in test_set:
        for value in test_set[label]:
            value = [v * np.pi / 180. for v in value]
            max_prob_density =  -sys.maxint - 1
            best_label = ""
            counter += 1
            for key in estimators:
                score = estimators[key].score(value)
                if score > max_prob_density:
                    max_prob_density = score
                    best_label = key
            print best_label, label
            if best_label == label:
                accurate +=1
    print accurate/counter * 100            




               
