import sys
import csv
import numpy as np
import math
import random
from sklearn.cross_validation import KFold
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.neighbors import DistanceMetric
from sklearn.grid_search import GridSearchCV
from sklearn.feature_selection import SelectFpr
from sklearn.feature_selection import chi2
from sklearn.preprocessing import StandardScaler
from spams.density_inference.perform_kde import priors_with_db_scan

import logging
logging.basicConfig(filename='classifier_motion.log',level=logging.DEBUG)

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

KERNEL_PARAMS = ['linear']
SVM_C_PARAMS = [0.01, 0.1, 1, 10]
params = {'svm__kernel': KERNEL_PARAMS, 'svm__C': SVM_C_PARAMS} #,  'selection__k': SELECTION_K_PARAMS}
KFOLDS = 10
PRIOR_WEIGHT = 0.01


def classify_top_level(x_train, y_train, x_test, y_priors=None):
    pipeline = Pipeline([('selection', SelectFpr(chi2, alpha=0.05)),('scaler', StandardScaler()),('svm', svm.SVC())])
    sample_weight = None
    if y_priors is not None:
        sample_weight = [1.0 for i in xrange(len(y_train))]
        y_train.extend(y_priors)
        x_train = np.vstack((x_train, x_test))
        sample_weight.extend([PRIOR_WEIGHT for i in xrange(len(y_priors))])
        clf = GridSearchCV(pipeline, params, fit_params={'svm__sample_weight' : sample_weight})
    else:
        clf = GridSearchCV(pipeline, params)
    clf.fit(x_train, y_train)
    clf = clf.best_estimator_
    logging.debug(clf)
    return clf.predict(x_test)

def train_classifier_and_predict(training, test, y_priors=None):
    if len(test) == 0:
        return 0, len(test)
    y_train, x_train = zip(*training) 
    y_test, x_test = zip(*test)
    sample_weight = None
    pipeline = Pipeline([('selection', SelectFpr(chi2, alpha=0.05)),('scaler', StandardScaler()),('svm', svm.SVC())])
    clf = GridSearchCV(pipeline, params)
    clf.fit(x_train, y_train)
    clf = clf.best_estimator_
    logging.debug(clf)
    result = [y == y_test[index] for index, y in enumerate(clf.predict(x_test))]
    return result.count(1), len(result)


def classify_other(training, test, y_priors=None):
    if len(test) == 0:
        return 0, len(test)
    y_train, x_train = zip(*training)
    y_test, x_test = zip(*test) 
    sample_weight = None
    y_training_other = [OTHER_MAPPING[y] for y in y_train]
    sports_training = [(y, x) for (y, x) in training if y in [6, 7]]
    shop_and_food_training = [(y, x) for (y, x) in training if y in [8, 9]]
    pipeline = Pipeline([('selection', SelectFpr(chi2, alpha=0.05)),('scaler', StandardScaler()),('svm', svm.SVC())])
    clf = GridSearchCV(pipeline, params)
    clf.fit(x_train, y_training_other)
    clf = clf.best_estimator_
    logging.debug(clf)
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
    count += c
    a,c  = train_classifier_and_predict(sports_training, sports_test)
    accurate += a
    count += c
    return accurate, count 

def top_level_accuracy(top_level_predictions, test_set):
    accurate = 0.0 
    for index, pred in enumerate(top_level_predictions):
         if pred == TOP_LEVEL_MAPPING[test_set[index][0]]:
             accurate += 1
    return accurate/len(top_level_predictions)         
             




def perform_multi_level_classification(places_features):
    X = []
    Y = []
    Z = []
    for place, user in places_features:
        label, features = places_features[(place, user)]
        X.append(features)
        Y.append(label)
        Z.append((place, user, label))
    X = np.array(X)
    Y = np.array(Y)
    Z = np.array(Z)
    n = Y.shape[0]
    kf = KFold(n=n, n_folds=KFOLDS)
    overall_accuracy = 0.0
    tla = 0.0
    home_accuracy = 0.0
    for train_index, test_index in kf:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        z_train, z_test = Z[train_index], Z[test_index]
        priors = priors_with_db_scan(z_test, z_train) 
        training_dataset = zip(y_train, X_train)
        home_training_dataset = [(y, x) for (y, x) in training_dataset if y in [1, 2]]
        work_training_dataset = [(y, x) for (y, x) in training_dataset if y in [3, 5]]
        other_training_dataset = [(y, x) for (y, x) in training_dataset if y in [4, 6, 7, 8, 9, 10]]
        test_set = zip(y_test, X_test)
        y_train_top_level = [TOP_LEVEL_MAPPING[y] for y in y_train]
        priors_top_level = [TOP_LEVEL_MAPPING[y] for y in priors]
        
        #priors_top_level = [random.randint(1,3) for y in priors]
        priors_top_level = None
        top_level_predictions = classify_top_level(X_train, y_train_top_level, X_test, priors_top_level)
        tla += top_level_accuracy(top_level_predictions, test_set)
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
        logging.debug((len(home_input), len(work_input), len(other_input)))        
        h_n, h_d = train_classifier_and_predict(home_training_dataset, home_input)
        home_accuracy += (h_n*1.0)/(h_d *1.0)
        w_n, w_d = train_classifier_and_predict(work_training_dataset, work_input)
        o_n, o_d = classify_other(other_training_dataset, other_input)
        overall_accuracy += ((h_n + w_n + o_n) * 1.0 )/ ((h_d + w_d + o_d) * 1.0)
    print tla/len(kf)
    print home_accuracy/len(kf)
    return overall_accuracy/ len(kf) 

if __name__ == "__main__":
    places_features = {}
    with open("features", "r") as f:
        reader = csv.reader(f)
        for row in reader:
            place = int(row[0])
            user = int(row[1])
            label = int(row[2])
            features = row[3:]
            features = [float(f) for f in features]
            places_features[(place, user)] = (label, features)
    with open("motion_features.csv", "r") as f:
        reader = csv.reader(f)
        for row in reader:
            place = int(row[0])
            user = int(row[1])
            label = int(row[2])
            features = row[3:]
            features = [math.fabs(float(f)) for f in features]
            for i, f in enumerate(features):
                if np.isnan(f):
                    features[i] = 0.0
            places_features[(place, user)][1].extend(features)

    
    
    accuracy = perform_multi_level_classification(places_features)
    print accuracy
    # with open("result_selection", "w") as f:
    #    writer = csv.writer(f)
    #    writer.writerow([accuracy])
