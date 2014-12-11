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


K = 10
def adaptive_kde(places_location, test, train):
    #nn = NearestNeighbors(n_neighbors=K, metric='haversine')
    #all_points = [(float(r[0]), float(r[1])) for r in connection.execute(select([places_location.c.latitude, places_location.c.longitude])).fetchall()]
    #all_points = np.array(all_points)
    #all_points_new = all_points * np.pi/180.
    #nn.fit(all_points_new)
    
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
        test_densities = []
        #print label
        for i, loc in enumerate(test_locations):
            #dist, indices = nn.kneighbors(loc)
            dist, indices = base_kernel.tree_.query(loc, k=min(25, len(training_set)))
            #print loc
            #print dist
            
            bandwidth = max(dist[0])
            #if int(bandwidth) == 0:
            #    print all_points[i]
            #print bandwidth
            if not bandwidth > 0.0:
                bandwidth = base_kernel.bandwidth
                print [l * (180 / np.pi) for l in loc]

            #print bandwidth
            k = KernelDensity(bandwidth = bandwidth, algorithm= base_kernel.algorithm, kernel=base_kernel.kernel, metric=base_kernel.metric, atol=base_kernel.atol, rtol=base_kernel.rtol,
                    breadth_first=base_kernel.breadth_first, leaf_size=base_kernel.leaf_size, metric_params=base_kernel.metric_params)
            k.fit(training_set)
            #print k.score(loc)
            test_densities.append(k.score(loc))
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
        
        #print (max_label, true_label)
        if max_label == true_label:
             accurate += 1 
        counter += 1 
    return accurate/counter


if __name__ == "__main__":
    places_location = get_table("places_location", metadata)
    a = 0.0
    for i in xrange(1):
        test, train = split_test_and_train(places_location)
        a += adaptive_kde(places_location, test, train)
    print a    

