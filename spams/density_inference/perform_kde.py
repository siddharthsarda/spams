import sys
import numpy as np
import math
from collections import defaultdict
from sklearn.neighbors import KernelDensity
from sqlalchemy.sql import select, and_
from sqlalchemy import func
from spams.db.utils import setup_database, get_table
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from spams.mappings import LABEL_PLACE_MAPPING
from sklearn.grid_search import GridSearchCV


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
    return accuracy, nrr/counter, missed

    
def train_kde(training_set):    
    xy = np.array(training_set)
    # Convert to radians
    xy *= np.pi /180.
    params = {'bandwidth': np.logspace(-3, 3, 20)}
    # do a grid search
    grid = GridSearchCV(KernelDensity(metric="haversine", algorithm="ball_tree"), params)
    grid.fit(xy)
    return grid.best_estimator_

def extract_places(places_location, label_id):
    lat_long_query = select([places_location.c.latitude, places_location.c.longitude]).where(places_location.c.place_label_int==label_id)
    return connection.execute(lat_long_query).fetchall()


def extract_visits(places_location, label_id):
    visits_10min = get_table("visits_10min", metadata)
    visits_with_places = places_location.join(visits_10min, onclause=and_(places_location.c.userid == visits_10min.c.userid, places_location.c.placeid == visits_10min.c.placeid)).alias("visits_with_places")
    query = select([visits_with_places.c.places_location_latitude, visits_with_places.c.places_location_longitude]).where(visits_with_places.c.places_location_place_label_int == label_id)
    return connection.execute(query).fetchall()

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

def perform_kde(places_location, input_func= extract_places):
    estimators = {}
    test_set_dict = {}
    training_set = []
    for label_id in xrange(1, 11):
        label = LABEL_PLACE_MAPPING[label_id]
        results = input_func(places_location, label_id)
        training_set, test_set = split_test_and_train(results)
        test_set_dict[label] = list(set(test_set))
        estimators[label] = train_kde(training_set)
    accuracy = test_kde(test_set_dict, estimators)
    print accuracy


def perform_kde_visits(places_location):
    perform_kde_visits(places_location, input_func=extract_visits)


def perform_kde_places(places_location):
    perform_kde_visits(places_location)


if __name__ == "__main__":
    places_location = get_table("places_location", metadata)
    min_lat, max_lat, min_long, max_long = connection.execute(select([func.min(places_location.c.latitude), func.max(places_location.c.latitude), func.min(places_location.c.longitude), func.max(places_location.c.longitude)])).fetchall()[0]
    # for i in xrange(1, 10):
    # perform_kde(places_location)
    # perform_kde(places_location, input_func=extract_visits)

