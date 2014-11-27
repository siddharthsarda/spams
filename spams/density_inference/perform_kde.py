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
from itertools import groupby, izip


metadata, connection = setup_database()

def test_kde(test_set, estimators):
    accurate = 0.0
    counter = 0.0
    nrr = 0.0
    missed = {}
    # Test accuracy using just kde
    for label in test_set:
        missed[label] = defaultdict(int)
        
        # print label
        # print len(test_set[label])
        for value in test_set[label]:
            # Convert to radians
            value = [v * np.pi / 180. for v in value]
            # print value
            best_label = ""
            counter += 1
            ## Loop over all estimators to find one with the highest density
            score_dict = {}
            for key in estimators:
                score_dict[key] = estimators[key].score(value)
            sorted_dict = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)    
            # print sorted_dict
            sorted_labels = [s[0] for s in sorted_dict]
            if sorted_labels[0] == label:
                accurate +=1
            else:
                missed[label][sorted_labels[0]] += 1

            nrr += (1.0/(1 + sorted_labels.index(label)))    
    accuracy = accurate/counter * 100
    # print missed
    return accurate, nrr, counter

    
def train_kde(training_set):    
    xy = np.array(training_set)
    # Convert to radians
    xy *= np.pi /180.
    params = {'bandwidth': np.logspace(-3, 3, 20)}
    # do a grid search
    grid = GridSearchCV(KernelDensity(metric="haversine", algorithm="ball_tree"), params)
    grid.fit(xy)
    # print grid.best_estimator_
    return grid.best_estimator_

def extract_places(places_location, label_id, restrict_to):
    lat_long_query = select([places_location.c.latitude, places_location.c.longitude]).where(places_location.c.place_label_int==label_id).where(places_location.c.id.in_(restrict_to))
    results = connection.execute(lat_long_query).fetchall()
    return [(float(r[0]), float(r[1])) for r in results]


def extract_visits(places_location, label_id, restrict_to):
    visits_10min = get_table("visits_10min", metadata)
    visits_with_places = places_location.join(visits_10min, onclause=and_(places_location.c.userid == visits_10min.c.userid, places_location.c.placeid == visits_10min.c.placeid)).alias("visits_with_places")
    query = select([visits_with_places.c.places_location_latitude, visits_with_places.c.places_location_longitude]).where(visits_with_places.c.places_location_place_label_int == label_id).where(visits_with_places.c.places_location_id.in_(restrict_to))
    results = connection.execute(query).fetchall()
    print len(results)
    return [(float(r[0]), float(r[1])) for r in results]

def split_test_and_train(places_location):
    test_dict = {}
    train_dict = {}
    for label in xrange(1, 11):
        label_places = select([places_location.c.id]).where(places_location.c.place_label_int==label)
        results = connection.execute(label_places).fetchall()
        results = [r[0] for r in results]
        result_len = len(results)
        test_len = math.ceil(result_len * 0.1)
        test_indices = np.random.choice(result_len, test_len)
        test_dict[label] = [results[i] for i in test_indices]
        train_dict[label] = [r for i,r in enumerate(results) if i not in test_indices]
    return test_dict, train_dict

def perform_kde(places_location, test, train, input_func= extract_places):
    estimators = {}
    test_set_dict = {}
    training_set = []
    for label_id in xrange(1, 11):
        label = LABEL_PLACE_MAPPING[label_id]
        training_set = input_func(places_location, label_id, train[label_id])
        estimators[label] = train_kde(training_set)
        test_set_dict[label] = [(float(r[0]), float(r[1])) for r in connection.execute(select([places_location.c.latitude, places_location.c.longitude]).where(places_location.c.id.in_(test[label_id]))).fetchall()]
    accuracy, mrr, test_set_size = test_kde(test_set_dict, estimators)
    #print accuracy
    return accuracy, mrr, test_set_size


def perform_kde_visits(places_location, restrict_to=None):
    return perform_kde(places_location, input_func=extract_visits, restrict_to=restrict_to)


def perform_kde_places(places_location, restrict_to=None):
    return perform_kde(places_location, restrict_to=restrict_to)


def perform_db_scan(places_location):
    lat_long_query = select([places_location.c.latitude, places_location.c.longitude, places_location.c.id])
    results = connection.execute(lat_long_query).fetchall()
    xy = np.asarray([(float(r[0]), float(r[1])) for r in results])
    places = [r[2] for r in results]
    db = DBSCAN(eps =0.025, metric="haversine", algorithm="ball_tree", min_samples=20).fit(xy)
    groups = db.labels_
    places_with_groups = izip(places, groups)
    place_group_dict = defaultdict(int)
    group_place_dict = defaultdict(list)
    for place, group in places_with_groups:
        place_group_dict[place] = group
        group_place_dict[group].append(place)
    return place_group_dict, group_place_dict

def kde_with_db_scan(places_location):
    pass

if __name__ == "__main__":
    places_location = get_table("places_location", metadata)
    #min_lat, max_lat, min_long, max_long = connection.execute(select([func.min(places_location.c.latitude), func.max(places_location.c.latitude), func.min(places_location.c.longitude), func.max(places_location.c.longitude)])).fetchall()[0]
    #print perform_kde_visits(places_location)
    #perform_db_scan(places_location)
    # place_group_dict, group_place_dict = perform_db_scan(places_location)
    overall_a = 0.0
    
    # accurate, nrr, counter = perform_kde(places_location,test, train, input_func=extract_visits)
    # print accurate, nrr, counter
    for i in xrange(10):
        test, train = split_test_and_train(places_location)
        accurate, nrr, counter = perform_kde(places_location,test, train, input_func=extract_visits)
        overall_a += (accurate/counter)
    print overall_a/10
    
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
