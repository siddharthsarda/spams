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

def test_kde(test_set, estimators):
    accurate = 0.0
    counter = 0.0
    # Test accuracy using just kde
    for label in test_set:
        for value in test_set[label]:
            # Convert to radians
            value = [v * np.pi / 180. for v in value]
            max_prob_density =  -sys.maxint - 1
            best_label = ""
            counter += 1
            ## Loop over all estimators to find one with the highest density
            for key in estimators:
                score = estimators[key].score(value)
                if score > max_prob_density:
                    max_prob_density = score
                    best_label = key
            #print max_prob_density
            if best_label == label:
                accurate +=1
    return accurate/counter * 100

    
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


def split_test_and_train(results):
    # Use 1 % of the data as test set
    result_len = len(results)
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


if __name__ == "__main__":
    metadata, connection = setup_database()
    places_location = get_table("places_location", metadata)
    min_lat, max_lat, min_long, max_long = connection.execute(select([func.min(places_location.c.latitude), func.max(places_location.c.latitude), func.min(places_location.c.longitude), func.max(places_location.c.longitude)])).fetchall()[0]
    estimators = {}
    test_set_dict = {}
    training_set = []
    for label_id in xrange(1, 11):
        label = LABEL_PLACE_MAPPING[label_id]
        results = extract_places(places_location, label_id)
        training_set, test_set = split_test_and_train(results)
        test_set_dict[label] = test_set
        estimators[label] = train_kde(training_set)
    accuracy = test_kde(test_set_dict, estimators)
    print accuracy
        
                




               
