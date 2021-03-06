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
from sklearn.feature_selection import chi2, f_classif
from sklearn.preprocessing import StandardScaler
from spams.density_inference.perform_kde import priors_with_kde
from collections import defaultdict
from spams.mappings import LABEL_PLACE_MAPPING

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
SVM_C_PARAMS = [0.01, 0.1]
ALPHA_PARAMS = [0.05]
params = {'svm__kernel': KERNEL_PARAMS, 'svm__C': SVM_C_PARAMS, 'selection__alpha': ALPHA_PARAMS} #,  'selection__k': SELECTION_K_PARAMS}
KFOLDS = 10
PRIOR_WEIGHT = 0.01
SELECTOR = chi2

ola = 0.0

def get_best_estimator(x_train, y_train, x_test, y_priors=None):
    pipeline = Pipeline([('selection', SelectFpr(SELECTOR)),('scaler', StandardScaler()),('svm', svm.SVC())])
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
    return clf


def classify_top_level(x_train, y_train, x_test, y_priors=None):
    clf = get_best_estimator(x_train, y_train, x_test, y_priors)
    return clf.predict(x_test)

def train_classifier_and_predict(training, test, use_priors=False, class_weight= None):
    if len(test) == 0:
        return 0, len(test), []
    y_train, x_train = zip(*training) 
    y_test, x_test = zip(*test)
    if use_priors:
        priors = priors_with_kde(y_test, y_train)
    else:
        priors = None
    
    places = [y[0] for y in y_test]
    y_test = [y[1] for y in y_test]
    y_train = [y[1] for y in y_train]
    clf = get_best_estimator(x_train, y_train, x_test, priors)
    logging.debug(clf)
    predictions = clf.predict(x_test)
    answers = zip(places, predictions, y_test)
    result = [y == y_test[index] for index, y in enumerate(predictions)]
    
    return result.count(1), len(result), answers


def extend_features(x_train, y_train, x_test, y_test, attribute, method='dbscan'):
    training_scores, test_scores = priors_with_kde(y_test, y_train, return_predictions=False, attribute = attribute)
    modified_training_set = []
    for y, x in zip(y_train, x_train):
        x_new = np.hstack((x, training_scores[y[0]]))
        modified_training_set.append(x_new)
    modified_test_set = []
    for y, x in zip(y_test, x_test):
        x_new = np.hstack((x, test_scores[y[0]]))
        modified_test_set.append(x_new)
    return modified_test_set, modified_training_set

def extend_features_with_split(train, test, attribute):
    y_train, x_train = zip(*train)
    y_test, x_test = zip(*test)
    training_scores, test_scores = priors_with_kde(y_test, y_train, return_predictions=False, attribute = attribute)
    modified_training_set = []
    for y, x in zip(y_train, x_train):
        x_new = np.hstack((x, training_scores[y[0]]))
        modified_training_set.append((y, x_new))
    modified_test_set = []
    for y, x in zip(y_test, x_test):
        x_new = np.hstack((x, test_scores[y[0]]))
        modified_test_set.append((y, x_new))
    return modified_test_set, modified_training_set
   


def other_level_accuracy(other_level_predictions, test_set):
    accurate = 0.0
    for index, pred in enumerate(other_level_predictions):
        try:
            if pred == OTHER_MAPPING[test_set[index][0][1]]:
                accurate += 1
        except KeyError:
            pass
    return accurate/len(other_level_predictions)

def classify_other(training, test, use_priors = False, add_features = False):
    if len(test) == 0:
        return 0, len(test), []
    y_train, x_train = zip(*training)
    y_test, x_test = zip(*test) 
    if use_priors:
        priors = priors_with_kde(y_test, y_train)
        priors_others = [OTHER_MAPPING[y] for y in priors]
    else:
        priors_others = None
    if add_features:
        x_test, x_train = extend_features(x_train, y_train, x_test, y_test, 'gender')
        #x_test, x_train = extend_features(x_train, y_train, x_test, y_test, 'working')
        #x_test, x_train = extend_features(x_train, y_train, x_test, y_test, 'age_group')
    
    #x_test, x_train = extend_features(x_train, y_train, x_test, y_test, 'label')
    y_training_other = [OTHER_MAPPING[y[1]] for y in y_train]
    result = classify_top_level(x_train, y_training_other, x_test, priors_others)
    global ola
    ola +=other_level_accuracy(result, test)
    accurate = 0.0
    count = 0.0
    answers = []
    
    sports_training = [(y, x) for (y, x) in training if y[1] in [6, 7]]
    shop_and_food_training = [(y, x) for (y, x) in training if y[1] in [8, 9]]
    sports_test = []
    food_shop_test = []

    for index, val in enumerate(result):
        if val == 0:
           sports_test.append(test[index])
        elif val == 1:
           food_shop_test.append(test[index])
        elif val == 2 or val == 3: 
           count += 1
           accurate += REVERSE_OUTER_MAPPING[val] == y_test[index][1]
           answers.append((y_test[index][0], REVERSE_OUTER_MAPPING[val], y_test[index][1]))

    #food_shop_test, shop_and_food_training = extend_features_with_split(shop_and_food_training, food_shop_test, 'age_group')

    #sports_test, sports_training = extend_features_with_split(sports_training, sports_test, 'age_group')
    a,c,d  = train_classifier_and_predict(shop_and_food_training, food_shop_test)
    accurate += a
    count += c
    answers.extend(d)
    a,c,d  = train_classifier_and_predict(sports_training, sports_test)
    accurate += a
    count += c
    answers.extend(d)
    return accurate, count, answers 

def top_level_accuracy(top_level_predictions, test_set):
    accurate = 0.0 
    for index, pred in enumerate(top_level_predictions):
         if pred == TOP_LEVEL_MAPPING[test_set[index][0][1]]:
             accurate += 1
    return accurate/len(top_level_predictions)         


def get_statistics(answers):
    each_label_accuracy = {}
    confusion_matrix = defaultdict(dict)
    for label in xrange(1, 11):
        each_label_accuracy[label] = [0, 0]
    for place, pred, label in answers:
        if LABEL_PLACE_MAPPING[pred] not in confusion_matrix[LABEL_PLACE_MAPPING[label]]:
            confusion_matrix[LABEL_PLACE_MAPPING[label]][LABEL_PLACE_MAPPING[pred]] = 0
        confusion_matrix[LABEL_PLACE_MAPPING[label]][LABEL_PLACE_MAPPING[pred]] += 1
        each_label_accuracy[label][1] += 1
        if pred == label:
            each_label_accuracy[label][0] += 1
    conf_items = sorted(confusion_matrix.items(), key=lambda x:x[0])
    num = 0.0
    denom = 0.0
    
    target = open("baseline.txt", 'r')
    for label in xrange(1, 11):
        cmatrix = []
        #if label in [4, 5, 6, 7, 8, 9, 10]:
        #    s = sum(confusion_matrix[LABEL_PLACE_MAPPING[label]].values())
        #    denom += s
        #    try:
        #        num += confusion_matrix[LABEL_PLACE_MAPPING[label]][LABEL_PLACE_MAPPING[label]]
        #    except:
        #        pass
        baseline_numbers = eval(target.readline())
        #print baseline_numbers
        s = sum(confusion_matrix[LABEL_PLACE_MAPPING[label]].values())
        for pred in xrange(1, 11):
            try:
                val = (confusion_matrix[LABEL_PLACE_MAPPING[label]][LABEL_PLACE_MAPPING[pred]] * 1.0)/s
                val  = math.ceil(val*10000)/100
                cmatrix.append(val - baseline_numbers[pred-1])
            except:
                cmatrix.append(0.0 - baseline_numbers[pred - 1])
        #target.write(str(cmatrix) + "\n")
        cmatrix = [str(c) for c in cmatrix]
        cmatrix[label - 1] = "\\textbf{" + cmatrix[label - 1] + "}"
        print LABEL_PLACE_MAPPING[label] + "&" +   "&".join(cmatrix) + "\\\\"    
    target.close()

    for k, v in conf_items:
        print k, [(key, (val*1.0)/sum(v.values())) for key, val in v.items()] 
    #print num/denom    
    return each_label_accuracy, confusion_matrix



def do_classification(X, Y, train_index, test_index, kde_as_priors = False, kde_as_features = False, add_features= False):
    answers = []
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]
    training_dataset = zip(y_train, X_train)
    test_set = zip(y_test, X_test)
    priors_top_level = None

    if kde_as_priors:
        priors = priors_with_kde(y_test, y_train)
        priors_top_level = [TOP_LEVEL_MAPPING[y] for y in priors]

    if kde_as_features:
        X_test, X_train = extend_features(X_train, y_train, X_test, y_test, 'label')
        test_set = zip(y_test, X_test)
        training_dataset = zip(y_train, X_train)
    #X_test, X_train = extend_features(X_train, y_train, X_test, y_test, 'time', method = 'simple')
    test_set = zip(y_test, X_test)
    training_dataset = zip(y_train, X_train)

        
    X_train = [x for (y, x) in training_dataset]
    X_test  = [x for (y, x) in test_set]
    home_training_dataset = [(y, x) for (y, x) in training_dataset if y[1] in [1, 2]]
    work_training_dataset = [(y, x) for (y, x) in training_dataset if y[1] in [3, 5]]
    other_training_dataset = [(y, x) for (y, x) in training_dataset if y[1] in [4, 6, 7, 8, 9, 10]]
    y_train_top_level = [TOP_LEVEL_MAPPING[y[1]] for y in y_train]
    top_level_predictions = classify_top_level(X_train, y_train_top_level, X_test, priors_top_level)
    tla = top_level_accuracy(top_level_predictions, test_set)
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
    h_n, h_d, home_answers = train_classifier_and_predict(home_training_dataset, home_input, use_priors = False)
    w_n, w_d, work_answers = train_classifier_and_predict(work_training_dataset, work_input, use_priors = False)
    o_n, o_d, other_answers = classify_other(other_training_dataset, other_input, use_priors=kde_as_priors, add_features = add_features)
    overall_accuracy = ((h_n + w_n + o_n) * 1.0 )/ ((h_d + w_d + o_d) * 1.0)
    for a in [home_answers, work_answers, other_answers]:
        answers.extend(a)
    return tla, overall_accuracy, answers


def perform_multi_level_classification(places_features):
    X = []
    Y = []
    for place, user in places_features:
        label, features = places_features[(place, user)]
        X.append(features)
        Y.append(((place, user) , label))
    X = np.array(X)
    Y = np.array(Y)
    n = Y.shape[0]
    kf = KFold(n=n, n_folds=KFOLDS)
    tla = 0.0
    all_answers = []
    overall_accuracy = 0.0
    X_records = []
    Y_records = []
    for train_index, test_index in kf:
       print "KFOLD"
       #_, _, baseline_answers =  do_classification(X, Y, train_index, test_index)
       tl, accuracy, answers = do_classification(X, Y, train_index, test_index, kde_as_priors = True, add_features = True)
       tla += tl
       overall_accuracy += accuracy
       all_answers.extend(answers)
       #baseline, algo = significance_test(baseline_answers, answers)
       #X_records.extend(baseline)
       #Y_records.extend(algo)
    #from scipy.stats import wilcoxon
    #print zip(X_records,Y_records)
    #print wilcoxon(X_records, Y_records)
    #gets confusion matrix and per label accuracy
    get_statistics(all_answers)
    print tla/len(kf)
    global ola
    print ola/len(kf)
    #print accuracy_other/ len(kf)
    return overall_accuracy/ len(kf)


def significance_test(baseline_answers, answers):
    baseline_dict = {}
    answer_dict = {}
    for place, pred, label in baseline_answers:
        baseline_dict[place] = pred
    for place, pred, label in answers:
        answer_dict[place] = pred
    X = []
    Y = []
    for key in baseline_dict:
        X.append(baseline_dict[key])
        Y.append(answer_dict[key])
    return X, Y

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
