import sys
import numpy as np
import math
from collections import defaultdict
from sklearn.neighbors import KernelDensity
from sqlalchemy.sql import select, and_
from sklearn.cluster import DBSCAN
from sqlalchemy import func
from spams.db.utils import setup_database, get_table
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from spams.mappings import LABEL_PLACE_MAPPING
from sklearn.grid_search import GridSearchCV
from itertools import groupby


metadata, connection = setup_database()

def test_kde(test_set, estimators):
    accurate = 0.0
    counter = 0.0
    nrr = 0.0
    missed = {}
    # Test accuracy using just kde
    for label in test_set:
        missed[label] = defaultdict(int)
        for value in test_set[label]:
            # Convert to radians
            value = [v * np.pi / 180. for v in value]
            best_label = ""
            counter += 1
            ## Loop over all estimators to find one with the highest density
            score_dict = {}
            for key in estimators:
                score_dict[key] = estimators[key].score(value)
            sorted_labels = [s[0] for s in sorted(score_dict.items(), key=lambda x: x[1], reverse=True)]
            if sorted_labels[0] == label:
                accurate +=1
            else:
                missed[label][sorted_labels[0]] += 1

            nrr += (1.0/(1 + sorted_labels.index(label)))    
    accuracy = accurate/counter * 100
    return accuracy, nrr/counter, counter

    
def train_kde(training_set):    
    xy = np.array(training_set)
    # Convert to radians
    xy *= np.pi /180.
    params = {'bandwidth': np.logspace(-5, 5, 20)}
    # do a grid search
    grid = GridSearchCV(KernelDensity(metric="haversine", algorithm="ball_tree"), params)
    grid.fit(xy)
    #print grid.best_estimator_
    return grid.best_estimator_

def extract_places(places_location, label_id, restrict_to=None):
    lat_long_query = select([places_location.c.latitude, places_location.c.longitude, places_location.c.userid, places_location.c.placeid]).where(places_location.c.place_label_int==label_id)
    results = connection.execute(lat_long_query).fetchall()
    if restrict_to:
        results = [r for r in results if (r[2], r[3]) in restrict_to]
    return results


def extract_visits(places_location, label_id):
    visits_10min = get_table("visits_10min", metadata)
    visits_with_places = places_location.join(visits_10min, onclause=and_(places_location.c.userid == visits_10min.c.userid, places_location.c.placeid == visits_10min.c.placeid)).alias("visits_with_places")
    query = select([visits_with_places.c.places_location_latitude, visits_with_places.c.places_location_longitude, visits_with_places.c.places_location_userid, visits_with_places.c.places_location_placeid]).where(visits_with_places.c.places_location_place_label_int == label_id)
    results = connection.execute(query).fetchall()
    if restrict_to:
        results = [r for r in results if (r[2], r[3]) in restrict_to]
    return results


def split_test_and_train(results):
    # Use 1 % of the data as test set
    result_len = len(results)
    # print result_len
    test_len = math.ceil(result_len * 0.1)
    xy = [(float(r[0]), float(r[1])) for r in results]
    test_indices = np.random.choice(result_len, test_len)
    training_set =[]
    test_set = []
    training_set = []
    # Separate test and training values
    for index, val in enumerate(xy):
        if index in test_indices:
            test_set.append(val)
        else:
            training_set.append(val)
    return training_set, test_set 

def perform_kde(places_location, input_func= extract_places, restrict_to=None):
    estimators = {}
    test_set_dict = {}
    training_set = []
    for label_id in xrange(1, 11):
        label = LABEL_PLACE_MAPPING[label_id]
        results = input_func(places_location, label_id, restrict_to)
        # print label
        if len(results) > 0:
            training_set, test_set = split_test_and_train(results)
            test_set_dict[label] = list(set(test_set))
            estimators[label] = train_kde(training_set)
    accuracy, mrr, test_set_size = test_kde(test_set_dict, estimators)
    #print accuracy
    return accuracy, mrr, test_set_size


def perform_kde_visits(places_location, restrict_to=None):
    return perform_kde(places_location, input_func=extract_visits, restrict_to=restrict_to)


def perform_kde_places(places_location, restrict_to=None):
    return perform_kde(places_location, restrict_to=restrict_to)


def perform_db_scan(places_location):
    lat_long_query = select([places_location.c.latitude, places_location.c.longitude, places_location.c.userid, places_location.c.placeid])
    results = connection.execute(lat_long_query).fetchall()
    xy = np.asarray([(float(r[0]), float(r[1])) for r in results])
    places = [(r[2], r[3]) for r in results]
    db = DBSCAN(eps =0.05, metric="haversine", algorithm="ball_tree", min_samples=2).fit(xy)
    labels = db.labels_
    indices_with_labels = list(enumerate(labels))
    
    def keyfunc(x):
        return x[1]

    indices_with_labels = sorted(indices_with_labels, key=keyfunc)
    index_label_dict = defaultdict(list)
    for key, group in groupby(indices_with_labels, keyfunc):
        index_label_dict[key].extend([g[0] for g in group])
    
    place_label_dict = {}
    for key in index_label_dict:
        place_label_dict[key] = [places[i] for i in index_label_dict[key]]
    
    for key in place_label_dict:
        if len(place_label_dict[key]) > 50:
            print perform_kde_places(places_location, restrict_to=place_label_dict[key])

    
    
if __name__ == "__main__":
    places_location = get_table("places_location", metadata)
    min_lat, max_lat, min_long, max_long = connection.execute(select([func.min(places_location.c.latitude), func.max(places_location.c.latitude), func.min(places_location.c.longitude), func.max(places_location.c.longitude)])).fetchall()[0]
    perform_db_scan(places_location)
    #for db_scan in (True, False):
    #    av = 0.0
    #    mrr = 0.0
    #    for i in xrange(1, 100):
    #        a, m, _ = perform_kde_places(places_location, with_db_scan=db_scan)
    #        av += a
    #        mrr += m
    #    print (av/100, mrr/100)
    
    #print perform_kde_visits(places_location, with_db_scan=True)
    #print perform_kde_visits(places_location, with_db_scan=False)
