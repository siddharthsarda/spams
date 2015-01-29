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
import operator


metadata, connection = setup_database()

best_global_bandwidths = {1: 0.000379269019073, 2: 0.0012742749857 , 3: 0.000379269019073, 4: 0.0012742749857, 5: 0.0012742749857,
                          6: 0.0012742749857, 7: 0.000379269019073, 8: 0.000379269019073, 9: 0.0012742749857, 10: 0.00428133239872}

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
                accurate += 1
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


def extract_places(places_location, label_id, restrict_to, attribute='label'):
    if len(restrict_to) == 0:
        return []
    if attribute == 'label':
        lat_long_query = select([places_location.c.latitude, places_location.c.longitude]).where(places_location.c.place_label_int==label_id).where(places_location.c.id.in_(restrict_to))
    
    else:
        demographics = get_table('demographics', metadata)
        joined_t = demographics.join(places_location, places_location.c.userid == demographics.c.userid)
        lat_long_query = select([places_location.c.latitude, places_location.c.longitude]).where(demographics.c[attribute]==label_id).where(places_location.c.id.in_(restrict_to)).select_from(joined_t)

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


def get_coordinates(places_location, p):
    lat_long_query = select([places_location.c.latitude, places_location.c.longitude]).where(places_location.c.id==p)
    val = [float(r) for r in connection.execute(lat_long_query).fetchall()[0]]
    val = [v* np.pi/180. for v in val]
    return val


def get_scores(places_location, p, predictor_iterator, estimators):
    scores = {}
    val = get_coordinates(places_location, p)
    for predictor in predictor_iterator:
        if predictor not in estimators:
            scores[predictor] = 0.0
            continue
        score = estimators[predictor].score(val)
        if score < 0:
            scores[predictor] = 0.0
        else: 
            scores[predictor] = score    
    #Sort by key    
    return sorted(scores.items(), key=operator.itemgetter(0))   


# train is place ids with label
# test is just place ids
def priors_with_db_scan(test, train, input_func=extract_places, attribute = 'label', return_predictions = True):
    #test is originally a tuple of place, user, label
    
    places_location = get_table("places_location", metadata)
    q = select([places_location.c.id]) 
    if attribute == 'label':
        train_tuples = [(connection.execute(q.where(and_(places_location.c.placeid == place, places_location.c.userid == user))).fetchall()[0][0], label) for ((place, user), label) in train]
        predictor_iterator = range(1, 11)
    else:
        demographics = get_table("demographics", metadata)
        demo_q = select([demographics.c[attribute]])
        train_tuples = []
        for (place, user), _ in train:
            id = connection.execute(q.where(and_(places_location.c.placeid == place, places_location.c.userid == user))).fetchall()[0][0]
            demo_r =  connection.execute(demo_q.where(demographics.c.userid == user))
            if demo_r.rowcount == 1:
                train_tuples.append((id, demo_r.fetchall()[0][0]))
        predictor_iterator = range(1, 3)

    test = [connection.execute(q.where(and_(places_location.c.placeid == place, places_location.c.userid == user))).fetchall()[0][0] for ((place, user), _) in test]
    train_predictor_dict = defaultdict(list)
    for id, predictor in train_tuples:
        train_predictor_dict[predictor].append(id)
    place_group_dict, group_place_dict = perform_db_scan(places_location)
    groups = group_place_dict.keys()
    test_scores = {}
    accurate = 0.0
    count = len(test)
    train_scores = {}
    for g in groups:
        estimators = {}
        group_test = [p for p in test if place_group_dict[p] == g]
        for predictor in predictor_iterator:
            places_in_group = [p for p in train_predictor_dict[predictor] if place_group_dict[p] == g]
            if len(places_in_group) == 0:
                continue
            training_set = input_func(places_location, predictor, places_in_group, attribute)
            xy = np.array(training_set)
            # Convert to radians
            xy *= np.pi /180.
            estimators[predictor] = train_kde(xy, predictor)

        for p in group_test:
            test_scores[p] = get_scores(places_location, p, predictor_iterator, estimators)
        for p in group_place_dict[g]:
            train_scores[p] = get_scores(places_location, p, predictor_iterator, estimators)
    accurate = 0.0
    count = 0.0
    for place in test_scores.keys():
        if attribute == 'label':
            true_predictor = connection.execute(select([places_location.c.place_label_int]).where(places_location.c.id==place)).fetchall()[0][0]
        else:
            user = connection.execute(select([places_location.c.userid]).where(places_location.c.id==place)).fetchall()[0][0]
            r = connection.execute(select([demographics.c[attribute]]).where(demographics.c.userid==user))
            if r.rowcount == 1:
                true_predictor = r.fetchall()[0][0]
            else:
                true_predictor = None
        
        predicted_value, max_score =  max(test_scores[place], key = lambda x: x[1])
        if predicted_value == true_predictor:
            accurate += 1
            #print max(scores[place])
        count += 1    
    accuracy = accurate/count
    if attribute is not 'label':
        print accuracy

    if return_predictions:
        prior_labels = []
        for place in test:
            predicted_value, max_score =  max(test_scores[place], key = lambda x: x[1])
            prior_labels.append(predicted_value)
        return prior_labels
    else:
        test_scores = {tuple(connection.execute(select([places_location.c.placeid, places_location.c.userid]).where(places_location.c.id == p)).fetchall()[0]): [s[1] for s in score] for p, score in test_scores.items()}
        training_scores = {tuple(connection.execute(select([places_location.c.placeid, places_location.c.userid]).where(places_location.c.id == p)).fetchall()[0]): [s[1] for s in score] for p, score in train_scores.items()}
        return training_scores, test_scores




if __name__ == "__main__":
    places_location = get_table("places_location", metadata)
    acc = 0.0
    nrr = 0.0
    for i in xrange(1000):
        test, train = split_test_and_train(places_location)
        #a, n, counter = perform_kde_places(places_location, test, train)
        a, n, counter = kde_with_db_scan(places_location, test, train)
        acc += a/counter
        nrr += n/counter
    print acc/1000, nrr/1000
