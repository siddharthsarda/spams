from perform_kde import *
from scipy.stats.mstats import gmean
from sklearn.neighbors import DistanceMetric
import math
from sklearn.neighbors.ball_tree import kernel_norm
from sklearn.neighbors import NearestNeighbors

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


K = 15
def adaptive_kde(places_location, test, train):
    nn = NearestNeighbors(n_neighbors=K, metric='haversine')
    all_points = [(float(r[0]), float(r[1])) for r in connection.execute(select([places_location.c.latitude, places_location.c.longitude])).fetchall()]
    all_points = np.array(all_points)
    all_points_new = all_points * np.pi/180.
    
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
    test_locations = np.array(test_locations)
    test_locations *= np.pi/180.

    train_place_ids = []
    for label, places in train.items():
        train_place_ids.extend(places)
    query = select([places_location.c.latitude, places_location.c.longitude, places_location.c.place_label]).where(places_location.c.id.in_(train_place_ids))
    results = connection.execute(query).fetchall()
    train_locations = [(float(r[0]), float(r[1])) for r in results]
    train_locations = np.array(train_locations)
    train_locations *= np.pi/180.
    nn.fit(train_locations)


     
    #print  nn.kneighbors(test_locations)[0].shape
    bandwidths = [max(dist) for dist in nn.kneighbors(test_locations)[0]]
    #print zip(bandwidths, test_place_labels)
    #sys.exit(0)
    for label_id in xrange(1, 11):
        if len(train[label_id]) == 0 or len(test[label_id]) == 0:
            continue
        label = LABEL_PLACE_MAPPING[label_id]
        training_set = extract_places(places_location, label_id, train[label_id])
        training_set = np.array(training_set)
        training_set *= np.pi/180.
        base_kernel = train_kde(training_set, label)
        test_densities = []
        for i, loc in enumerate(test_locations):
            #dist, indices = nn.kneighbors(loc)
            dist, indices = base_kernel.tree_.query(loc, k=min(K, len(training_set)))
            
            #bandwidth = bandwidths[i]
            bandwidth = max(dist[0])
            #print (bandwidth, base_kernel.bandwidth)
            if not bandwidth > 0.0:
                bandwidth = base_kernel.bandwidth
                #print [l * (180 / np.pi) for l in loc]

            k = KernelDensity(bandwidth = bandwidth, algorithm= base_kernel.algorithm, kernel=base_kernel.kernel, metric=base_kernel.metric, atol=base_kernel.atol, rtol=base_kernel.rtol,
                    breadth_first=base_kernel.breadth_first, leaf_size=base_kernel.leaf_size, metric_params=base_kernel.metric_params)
            k.fit(training_set)
            test_densities.append(k.score(loc))
        estimator_densities[label_id] = test_densities
    accurate = 0.0
    counter = 0.0
    nrr = 0.0
    for index, true_label in enumerate(test_place_labels):
        max_label = None
        max_density = -sys.maxint - 1
        densities ={}
        for label_id in xrange(1, 11):
            label = LABEL_PLACE_MAPPING[label_id]
            densities[label] = estimator_densities[label_id][index]
        densities = sorted(densities.items(), key=lambda x: x[1], reverse=True)
        predicted_labels = [label for label,_ in densities]
        if predicted_labels[0] == true_label:
             accurate += 1
        nrr += (1.0/(predicted_labels.index(true_label) + 1)) 
        #print (predicted_labels[0], true_label) 
        counter += 1 
    return accurate/counter, nrr/counter


if __name__ == "__main__":
    places_location = get_table("places_location", metadata)
    net_a = 0.0
    net_n = 0.0
    for i in xrange(1000):
        test, train = split_test_and_train(places_location)
        a, n = adaptive_kde(places_location, test, train)
        net_a += a
        net_n += n
    print net_a/1000, net_n/1000

