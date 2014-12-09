from perform_kde import *
from scipy.stats.mstats import gmean
from sklearn.neighbors import DistanceMetric
import math
from sklearn.neighbors.ball_tree import kernel_norm


def gaussian_density(dist, h):
    return np.exp(-0.5 * (dist * dist) / (h * h))


def exponential_density(dist, h):
    return np.exp(-dist/h)

def exponential_norm(h,dimension=2):
    return kernel_norm(h, dimension, "exponential")

def haversine_distance(p1, p2):
    d = DistanceMetric.get_metric('haversine')
    X = [p1, p2]
    return d.pairwise(X)[0][1]



def adaptive_kde(places_location, test, train):
    estimator_densities = {}
    test_set_dict = {}
    training_set = []
    test_place_ids = []
    for label, places in test.items():
        test_place_ids.extend(places)
    query = select([places_location.c.latitude, places_location.c.longitude, places_location.c.place_label]).where(places_location.c.id.in_(test_place_ids))
    results = connection.execute(query).fetchall()
    test_locations = [(float(r[0]), float(r[1])) for r in results]
    test_place_labels = [r[2] for r in results]
    print len(test_locations)
    test_locations = np.array(test_locations)
    test_locations *= np.pi/180.
    # print test_locations
    for label_id in xrange(1, 11):
        if len(train[label_id]) == 0 or len(test[label_id]) == 0:
            continue
        label = LABEL_PLACE_MAPPING[label_id]
        #print label
        training_set = extract_places(places_location, label_id, train[label_id])
        training_set = np.array(training_set)
        training_set *= np.pi/180.
        base_kernel = train_kde(training_set, label)
        # Calculate exponential of the log of densities sent
        densities = [np.exp(x) for x in base_kernel.score_samples(training_set)]
        #print densities
        ## Get geometric mean of densities G
        density_mean = gmean(densities)
        h = base_kernel.bandwidth
        ## calculate local bandwidths for each x (h_i = h * (G/f(x_i))^0.5)
        local_bandwidths = [h * math.sqrt(density_mean/d) for d in densities]
        #print local_bandwidths
        test_densities = []
        for loc in test_locations:
            dens = 0.0
            for index, point in enumerate(training_set):
                d = haversine_distance(loc, point)
                b = local_bandwidths[index]
                dens += (exponential_density(d, b)/ b*b)
            log_dens = np.log(dens/len(training_set))
            test_densities.append(log_dens)
        estimator_densities[label_id] = test_densities
        #print estimator_densities[label_id]
    accurate = 0.0
    counter = 0.0
    for index, true_label in enumerate(test_place_labels):
        max_label = None
        max_density = -sys.maxint - 1
        for label_id in xrange(1, 11):
            if estimator_densities[label_id][index] > max_density:
                max_label = LABEL_PLACE_MAPPING[label_id]
                max_density = estimator_densities[label_id][index]
        
        print (max_label, true_label)
        if max_label == true_label:
             accurate += 1 
        counter += 1 
    print accurate/counter


if __name__ == "__main__":
    places_location = get_table("places_location", metadata)
    test, train = split_test_and_train(places_location)
    adaptive_kde(places_location, test, train)

