import sys
import csv
import datetime
import math
from spams.db.utils import setup_database, get_table
from sqlalchemy.sql import select
from sqlalchemy import and_, func
from sklearn.neighbors import DistanceMetric
import numpy as np
from itertools import izip
from extract_features import write_features_to_csv

import logging
logging.basicConfig(filename='motion.log',level=logging.DEBUG)


metadata, connection = setup_database()
places_location = get_table("places_location", metadata)
visits_10min = get_table("visits_10min", metadata)
records = get_table("records", metadata)
accel = get_table("accel", metadata)

def get_features(data):
    X = [d[0] for d in data] 
    Y = [d[1] for d in data]
    Z = [d[2] for d in data]
    x_mean = np.mean(X)
    y_mean = np.mean(Y)
    z_mean = np.mean(Z)
    x_var  = np.var(X)
    y_var =  np.var(Y)
    z_var =  np.var(Z)
    mean_magnitude = np.mean([math.sqrt(x*x + y*y +z*z) for (x,y,z) in izip(X,Y,Z)]) 
    magnitude_mean = math.sqrt(x_mean*x_mean + y_mean*y_mean + z_mean*z_mean)
    sma = np.mean([math.fabs(x) + math.fabs(y) + math.fabs(z) for (x,y,z) in izip(X,Y,Z)])
    corr_xy = (np.cov(X,Y) / (math.sqrt(x_var) * math.sqrt(y_var)))[0][1]
    corr_yz = (np.cov(Y,Z) / (math.sqrt(z_var) * math.sqrt(y_var)))[0][1]
    corr_xz = (np.cov(Z,X) / (math.sqrt(x_var) * math.sqrt(z_var)))[0][1]
    vector_d = [(x - x_mean, y - y_mean, z - z_mean) for (x,y,z) in izip(X,Y,Z)]
    vector_v = [x_mean, y_mean, z_mean]
    vector_p = [np.multiply((np.dot(d, vector_v)/np.dot(vector_v, vector_v)), vector_v) for d in vector_d]
    
    vector_h = [np.subtract(d, p) for d, p in izip(vector_d, vector_p)]
    mod_vector_p = [np.linalg.norm(p) for p in vector_p]
    mod_vector_h = [np.linalg.norm(h) for h in vector_h]
    cor_p_h = (np.cov(mod_vector_h,mod_vector_p) / (math.sqrt(np.var(mod_vector_h)) * math.sqrt(np.var(mod_vector_p))))[0][1]
    
    vector_p = np.mean(vector_p, axis=0)
    vector_h = np.mean(vector_h, axis=0)
    mod_vector_p = np.mean(mod_vector_p)
    mod_vector_h = np.mean(mod_vector_h)
    ret = [x_mean, y_mean, z_mean, x_var, y_var, z_var, mean_magnitude, magnitude_mean, sma, corr_xy, corr_yz, corr_xz, cor_p_h, mod_vector_p, mod_vector_h]
    ret.extend([x for x in vector_p])

    ret.extend([x for x in vector_h])
    return ret

if __name__ == "__main__":
    query = select([places_location.c.placeid, places_location.c.userid, places_location.c.place_label_int])
    places_with_label = [(r[0], r[1], r[2]) for r in connection.execute(query).fetchall()]
    place_label_features = {}
    for i, (place, user, label) in enumerate(places_with_label):
        visits = [(r[0], r[1]) for r in connection.execute(select([visits_10min.c.time_start, visits_10min.c.time_end]).where(and_(visits_10min.c.placeid == place, visits_10min.c.userid == user))).fetchall()]
        accel_data = []
        visit_features = []
        for start, end in visits:
            single_visit_features = []
            q = select([records.c.db_key]).where(and_(records.c.userid == user, records.c.type == "accel", records.c.time>=start, records.c.time <=end))
            accelerometer_records = connection.execute(q).fetchall()
            avdels = []
            if len(accelerometer_records) == 0:
                continue
            for r in accelerometer_records:
                key = r[0]
                query = select([accel.c.data, accel.c.avdelt]).where(and_(accel.c.db_key == key))
                d, avdel = connection.execute(query).fetchall()[0]
                avdels.append(avdel)
                import zlib
                import base64
                data = zlib.decompress(base64.decodestring(d))
                try:
                    values = eval(data)
                except Exception as e:
                    data = zlib.decompress(base64.decodestring(data))
                    values = eval(data)
                if len(values) < 5:
                    continue
                features = get_features([(x, y, z) for (t, x, y, z) in values])
                single_visit_features.append(features)
                #accel_data.extend([(x, y, z) for (t, x, y, z) in values])
            logging.debug((place, user))
            if len(single_visit_features) != 0:
                sf = np.mean(single_visit_features, axis=0)
                sf = list(sf)
                sf.extend([np.mean(avdels), np.var(avdels)])
                visit_features.append(sf)
            
            #visit_features = np.append(visit_features, [np.mean(avdels), np.var(avdels)])
        if len(visit_features) == 0:
            #logging.debug((place, user))
            average_features = [0.0 for i in xrange(23)]
        else:
            average_features = np.mean(visit_features, axis=0)
        place_label_features[(place, user)] = (label, average_features)
        logging.debug(i)
    write_features_to_csv("motion_all_features.csv", place_label_features)    
