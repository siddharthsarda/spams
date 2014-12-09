import sys
import numpy as np
import math
from collections import defaultdict
from sklearn.neighbors import KernelDensity
from sqlalchemy.sql import select, and_
from sklearn.cluster import DBSCAN
from sqlalchemy import func
from spams.db.utils import setup_database, get_table
# from mpl_toolkits.basemap import Basemap
# import matplotlib.pyplot as plt
from spams.mappings import LABEL_PLACE_MAPPING
from sklearn.grid_search import GridSearchCV
from itertools import groupby, izip, product


metadata, connection = setup_database()

best_global_bandwidths = {'Home': 0.000379269019073, 'Home of a Friend': 0.0012742749857 , 'Work': 0.000379269019073, 'Transport Related': 0.0012742749857, 'Work of a Friend': 0.0012742749857,
                          'Indoor Sports': 0.0012742749857, 'Outdoor Sports': 0.000379269019073, 'Bar;Restaurant': 0.000379269019073, 'Shop': 0.0012742749857, 'Holidays Resort': 0.00428133239872}

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
            sorted_dict = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)    
            sorted_labels = [s[0] for s in sorted_dict]
            if sorted_labels[0] == label:
                accurate +=1
            else:
                missed[label][sorted_labels[0]] += 1

            nrr += (1.0/(1 + sorted_labels.index(label)))    
    if counter:
        accuracy = accurate/counter * 100
    return accurate, nrr, counter

    
def train_kde(xy, label):    
    params = {'bandwidth': np.logspace(-5, 5, 20), 'kernel' : ['gaussian','exponential']}
    # do a grid search
    try:
        grid = GridSearchCV(KernelDensity(metric="haversine", algorithm="ball_tree"), params)
        grid.fit(xy)
        return grid.best_estimator_
    except ValueError:
        k = KernelDensity(metric="haversine", algorithm="ball_tree", bandwidth=best_global_bandwidths[label], kernel="exponential")
        k.fit(xy)
        return k


def extract_places(places_location, label_id, restrict_to):
    if len(restrict_to) == 0:
        return []

    lat_long_query = select([places_location.c.latitude, places_location.c.longitude]).where(places_location.c.place_label_int==label_id).where(places_location.c.id.in_(restrict_to))
    results = connection.execute(lat_long_query).fetchall()
    return [(float(r[0]), float(r[1])) for r in results]


def extract_visits(places_location, label_id, restrict_to):
    if len(restrict_to) == 0:
        return []
    visits_10min = get_table("visits_10min", metadata)
    visits_with_places = places_location.join(visits_10min, onclause=and_(places_location.c.userid == visits_10min.c.userid, places_location.c.placeid == visits_10min.c.placeid)).alias("visits_with_places")
    query = select([visits_with_places.c.places_location_latitude, visits_with_places.c.places_location_longitude]).where(visits_with_places.c.places_location_place_label_int == label_id).where(visits_with_places.c.places_location_id.in_(restrict_to))
    results = connection.execute(query).fetchall()
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
        test_indices = np.random.choice(result_len, test_len, replace=False)
        test_dict[label] = [results[i] for i in test_indices]
        train_dict[label] = [r for i,r in enumerate(results) if i not in test_indices]
    return test_dict, train_dict

def perform_kde(places_location, test, train, input_func= extract_places):
    estimators = {}
    test_set_dict = {}
    training_set = []
    for label_id in xrange(1, 11):
        if len(train[label_id]) == 0 or len(test[label_id]) == 0:
            continue
        label = LABEL_PLACE_MAPPING[label_id]
        training_set = input_func(places_location, label_id, train[label_id])
        xy = np.array(training_set)
        # Convert to radians
        xy *= np.pi /180.
        estimators[label] = train_kde(xy, label)
        test_set_dict[label] = [(float(r[0]), float(r[1])) for r in connection.execute(select([places_location.c.latitude, places_location.c.longitude]).where(places_location.c.id.in_(test[label_id]))).fetchall()]
    accuracy, mrr, test_set_size = test_kde(test_set_dict, estimators)
    return accuracy, mrr, test_set_size


def perform_kde_visits(places_location, test, train, restrict_to=None):
    return perform_kde(places_location, test, train, input_func=extract_visits)


def perform_kde_places(places_location, test, train):
    return perform_kde(places_location, test, train, input_func=extract_places)


def perform_db_scan(places_location, eps=0.02, min_samples=30):
    lat_long_query = select([places_location.c.latitude, places_location.c.longitude, places_location.c.id])
    results = connection.execute(lat_long_query).fetchall()
    xy = np.asarray([(float(r[0]), float(r[1])) for r in results])
    places = [r[2] for r in results]
    db = DBSCAN(eps, metric="haversine", algorithm="ball_tree", min_samples = min_samples).fit(xy)
    groups = db.labels_
    places_with_groups = izip(places, groups)
    place_group_dict = defaultdict(int)
    group_place_dict = defaultdict(list)
    for place, group in places_with_groups:
        place_group_dict[place] = group
        group_place_dict[group].append(place)
    return place_group_dict, group_place_dict

def kde_with_db_scan(places_location, test, train, input_func=extract_places):
    place_group_dict, group_place_dict = perform_db_scan(places_location)
    groups = group_place_dict.keys()
    group_train_dict = {}
    group_test_dict = {}
    for g in groups:
        group_train_dict[g] = {}
        group_test_dict[g] = {}
        for label in train.keys():
            group_train_dict[g][label] = [p for p in train[label] if place_group_dict[p] == g]
            group_test_dict[g][label] = [p for p in test[label] if place_group_dict[p] == g]
        
    net_acc = 0.0
    net_nrr = 0.0
    net_counter = 0.0
    for g in groups:
        accurate, nrr, counter = perform_kde(places_location, group_test_dict[g], group_train_dict[g], input_func=input_func)
        net_acc += accurate
        net_nrr += nrr
        net_counter += counter
    return net_acc, net_nrr, net_counter


if __name__ == "__main__":
    places_location = get_table("places_location", metadata)

    acc = 0.0
    nrr = 0.0
    for i in xrange(1):
        test, train = split_test_and_train(places_location)
        #a, n, counter = perform_kde_places(places_location, test, train)
        a, n, counter = kde_with_db_scan(places_location, test, train)
        acc += a/counter
        nrr += n/counter
    print acc, nrr
