import sys
import numpy as np
import math
from spams.db.utils import setup_database, get_table
from sqlalchemy.sql import select
from sklearn.cross_validation import KFold
from sklearn import svm

metadata, connection = setup_database()

TOP_LEVEL_MAPPING = {
    2: 0,  # "Home of a friend",
    6: 2,  # "Indoor sports",
    3: 1,  # "Work",
    8: 2,  #"Bar;Restaurant",
    7: 2,  # "Outdoor sports",
    10: 2, # "Holidays resort",
    1: 0, #"Home",
    4: 2, #"Transport related",
    5: 1, #"Work of a friend",
    9: 2 #"Shop"
}

OTHER_MAPPING = {
    6: 0,  # "Indoor sports",
    7: 0,  # "Outdoor sports",
    8: 1,  # Bar;Restaurant
    9: 1,   # Shop
    4: 2, #Transport related
    10 : 3 # Outdoors
}

REVERSE_OUTER_MAPPING = {val:key for (key,val) in OTHER_MAPPING.items()}


def extract_features(place_id, user_id):
    return np.random.rand(1,3)[0]

def classify_top_level(x_train, y_train, x_test):
    clf = svm.SVC()
    clf.fit(x_train, y_train)
    return clf.predict(x_test)

def train_classifier_and_predict(training, test):
    if len(training) == 0 or len(test) == 0:
        return 0,0
    y_train, x_train = zip(*training) 
    y_test, x_test = zip(*test) 
    clf = svm.SVC()
    clf.fit(x_train, y_train)
    result = [y == y_test[index] for index, y in enumerate(clf.predict(x_test))]
    return result.count(1), len(result)

def classify_other(training, test):
    if len(training) == 0 or len(test) == 0:
        return 0,0
    y_train, x_train = zip(*training)
    y_training_other = [OTHER_MAPPING[y] for y in train]
    sports_training = [(y, x) for (y, x) in training if y in [6, 7]]
    shop_and_food_training = [(y, x) for (y, x) in training if y in [8, 9]]
    y_test, x_test = zip(*test) 
    clf = svm.SVC()
    clf.fit(x_train, y_training_other)
    result = clf.predict(x_test)
    accurate = 0.0
    count = 0.0
    sports_test = []
    food_shop_test = []
    for index, val in enumerate(result):
        if val == 0:
           sports_test.append(test[index])
        elif val == 1:
           food_shop_test.append(test[index])
        elif val == 2 or val == 3: 
           count += 1
           accurate += REVERSE_OUTER_MAPPING[val] == y_test[index]
    a,c  = train_classifier_and_predict (shop_and_food_training, food_shop_test)
    accurate += a
    count += a
    a,c  = train_classifier_and_predict(sports_training, sports_test)
    accurate += a
    count += a
    return accurate, count 


def perform_multi_level_classification(places_features):
    X = []
    Y = []
    for label, features in places_features.values():
        X.append(features)
        Y.append(label)
    X = np.array(X)
    Y = np.array(Y)
    n = Y.shape[0]
    kf = KFold(n=n, n_folds=10)
    for train_index, test_index in kf:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        training_dataset = zip(y_train, X_train)
        home_training_dataset = [(y, x) for (y, x) in training_dataset if y in [1, 2]]
        work_training_dataset = [(y, x) for (y, x) in training_dataset if y in [3, 4]]
        other_training_dataset = [(y, x) for (y, x) in training_dataset if y in xrange(5, 10)]
        test_set = zip(y_test, X_test)
        y_train_top_level = [TOP_LEVEL_MAPPING[y] for y in y_train]
        top_level_predictions = classify_top_level(X_train, y_train_top_level, X_test)
        home_input = []
        work_input = []
        other_input = []

        for index, pred in enumerate(top_level_predictions):
            if pred == 0:
                home_input.append(test_set[index])
            elif pred == 1:
                work_input.append(test_set[index])
            else:
                other_input.append(test_set[index])
        train_classifier_and_predict(home_training_dataset, home_input)
        train_classifier_and_predict(work_training_dataset, work_input)
        classify_other(other_training_dataset, other_input)

        #for index, val in enumerate(top_level_predictions):








if __name__ == "__main__":
    places_location = get_table("places_location", metadata)
    query = select([places_location.c.placeid, places_location.c.userid, places_location.c.place_label_int])
    places_with_label = ((r[0], r[1], r[2]) for r in connection.execute(query).fetchall())
    places_features = {(place,user): (label, extract_features(place, user)) for (place,user,label) in places_with_label}
    perform_multi_level_classification(places_features)
