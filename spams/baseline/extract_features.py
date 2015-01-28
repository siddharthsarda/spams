import sys
import csv
import datetime
import math
from spams.db.utils import setup_database, get_table
from sqlalchemy.sql import select
from sqlalchemy import and_, func
from sklearn.neighbors import DistanceMetric


metadata, connection = setup_database()
places_location = get_table("places_location", metadata)
visits_10min = get_table("visits_10min", metadata)
records = get_table("records", metadata)



def relative_frequency(place, user):
    count_place =  connection.execute(select([visits_10min.c.userid, visits_10min.c.placeid]).where(and_(visits_10min.c.userid == user, visits_10min.c.placeid == place))).rowcount
    count_all_places = connection.execute(select([visits_10min.c.userid]).where(visits_10min.c.userid == user)).rowcount 
    return (count_place * 1.0) / (count_all_places * 1.0)

def distance_from_most_visited_place(place, user):
    q = select([func.count(),visits_10min.c.placeid]).where(visits_10min.c.userid == user).group_by(visits_10min.c.placeid).order_by(func.count().desc())
    most_visited_places = [r[1] for r in connection.execute(q).fetchall()]
    def get_lat_long(place_q):
        try:
            return connection.execute(select([places_location.c.longitude, places_location.c.latitude]).where(and_(places_location.c.placeid == place_q, places_location.c.userid == user))).fetchall()[0]
        except Exception as e:
            return None
            
    dist = DistanceMetric.get_metric('haversine')
    X = []
    X.append(get_lat_long(place))
    for p in most_visited_places:
        ret = get_lat_long(p)
        if ret is not None:
            X.append((ret[0], ret[1]))
            break
    return dist.pairwise(X)[0][1]

def calendar_time_frequency(place, user):
    calendar = get_table("calendar", metadata)
    visit_times = select([visits_10min.c.time_start]).where(and_(visits_10min.c.userid == user, visits_10min.c.placeid == place)) 
    visit_times = [t[0] for t in connection.execute(visit_times).fetchall()]
    freq = 0.0
    for t in visit_times:
        user_records = select([records.c.db_key]).where(and_(records.c.userid == user, records.c.time >= t - 300, records.c.time<= t+300, records.c.type == 'calendar')).alias("user_records")
        if connection.execute(user_records).rowcount != 0:
            freq += 1
    return freq        

def application_frequency_and_variety(place, user):
    application = get_table("application", metadata)
    visit_times = select([visits_10min.c.time_start, visits_10min.c.time_end]).where(and_(visits_10min.c.userid == user, visits_10min.c.placeid == place)) 
    visit_times = [(t[0], t[1]) for t in connection.execute(visit_times).fetchall()]
    average_freq = 0.0
    applications = set()
    user_records = [(r[0], r[1]) for r in connection.execute(select([records.c.db_key, records.c.time]).where(and_(records.c.userid == user,records.c.type == 'app'))).fetchall()]
    for start, end in visit_times:
        filtered_keys = [r[0] for r in user_records if r[1] >= start and r[1] <=end]
        if len(filtered_keys) == 0:
            continue
        calendar_entry_for_user = select([func.count(application.c.uid), application.c.uid]).where(application.c.db_key.in_(filtered_keys)).group_by(application.c.uid)
        times_used = 0.0
        for r, uid in connection.execute(calendar_entry_for_user).fetchall():
            times_used += r
            applications.add(uid)
        average_freq += (times_used / ((end-start)/3600.0))
    return average_freq/len(visit_times), len(applications)

def process_usage_frequency(place, user):
    process = get_table("processrelation", metadata)
    visit_times = select([visits_10min.c.time_start, visits_10min.c.time_end]).where(and_(visits_10min.c.userid == user, visits_10min.c.placeid == place)) 
    visit_times = [(t[0], t[1]) for t in connection.execute(visit_times).fetchall()]
    average_freq = 0.0
    applications = set()
    
    user_records = [(r[0], r[1]) for r in connection.execute(select([records.c.db_key, records.c.time]).where(and_(records.c.userid == user,records.c.type == 'process'))).fetchall()]
    for start, end in visit_times:
        filtered_keys = [r[0] for r in user_records if r[1] >= start and r[1] <=end]
        if len(filtered_keys) == 0:
            continue
        calendar_entry_for_user = select([func.count(process.c.pathid), process.c.pathid]).where(process.c.db_key.in_(filtered_keys)).group_by(process.c.pathid)
        times_used = 0.0
        for r, uid in connection.execute(calendar_entry_for_user).fetchall():
            times_used += r
            applications.add(uid)
        average_freq += (times_used / ((end-start)/3600.0))
    return average_freq/len(visit_times), len(applications)

def media_usage_frequency(place, user):
    mediaplay = get_table("mediaplay", metadata)
    visit_times = select([visits_10min.c.time_start, visits_10min.c.time_end]).where(and_(visits_10min.c.userid == user, visits_10min.c.placeid == place)) 
    visit_times = [(t[0], t[1]) for t in connection.execute(visit_times).fetchall()]
    average_freq = 0.0
    user_records = [(r[0], r[1]) for r in connection.execute(select([records.c.db_key, records.c.time]).where(and_(records.c.userid == user,records.c.type == 'media_play'))).fetchall()]
    if len(user_records) == 0:
        return 0
    for start, end in visit_times:
        filtered = [r[0] for r in user_records if r[1] >= start and r[1] <=end]
        times_used =  len(filtered)
        average_freq += (times_used / ((end-start)/3600.0))
    return average_freq

def communication_frequency(place, user):
    calllog = get_table("calllog", metadata)
    visit_times = select([visits_10min.c.time_start, visits_10min.c.time_end]).where(and_(visits_10min.c.userid == user, visits_10min.c.placeid == place)) 
    visit_times = [(t[0], t[1]) for t in connection.execute(visit_times).fetchall()]
    callout = missed_call = textout = 0.0
    user_records = [(r[0], r[1]) for r in connection.execute(select([records.c.db_key, records.c.time]).where(and_(records.c.userid == user,records.c.type == 'call_log'))).fetchall()]
    if len(user_records) == 0:
        return 0, 0, 0
    num_visits = len(visit_times)
    for start, end in visit_times:
        filtered = [r[0] for r in user_records if r[1] >= start and r[1] <=end]
        if(len(filtered)) == 0:
            continue
        stay_time = (end-start)/3600.0
        callout += (connection.execute(select([func.count(calllog.c.db_key)]).where(and_(calllog.c.description == "Voice call", calllog.c.direction == 'Outgoing', calllog.c.db_key.in_(filtered)))).fetchall()[0][0])/stay_time
        missed_call += (connection.execute(select([func.count(calllog.c.db_key)]).where(and_(calllog.c.description == "Voice call", calllog.c.direction == 'Missed call', calllog.c.db_key.in_(filtered)))).fetchall()[0][0])/stay_time
        textout += (connection.execute(select([func.count(calllog.c.db_key)]).where(and_(calllog.c.description == "Short message", calllog.c.direction == 'Outgoing', calllog.c.db_key.in_(filtered)))).fetchall()[0][0])/stay_time
    return (callout/num_visits, missed_call/num_visits, textout/num_visits)


def average_stay_time(place, user):
    visit_times = select([visits_10min.c.time_start, visits_10min.c.time_end]).where(and_(visits_10min.c.userid == user, visits_10min.c.placeid == place))
    visit_times = [(t[0], t[1]) for t in connection.execute(visit_times).fetchall()]
    total_time = 0.0
    for start, end in visit_times:
        total_time += ((end-start)/3600.0)
    return total_time/len(visit_times)

def holiday_to_weekday(place, user):
    visit_times = select([visits_10min.c.time_start, visits_10min.c.time_end]).where(and_(visits_10min.c.userid == user, visits_10min.c.placeid == place))
    visit_times = [(t[0], t[1]) for t in connection.execute(visit_times).fetchall()]
    holiday = 0.0
    weekday = 1.0
    for start, end in visit_times:
        if datetime.datetime.fromtimestamp(start).weekday() in [5,6]:
            holiday += 1
        else:
            weekday += 1
    return holiday/weekday

def time_features(place, user):
    visit_times = select([visits_10min.c.time_start, visits_10min.c.time_end, visits_10min.c.tz_start, visits_10min.c.tz_end]).where(and_(visits_10min.c.userid == user, visits_10min.c.placeid == place))
    visit_times = [(t[0], t[1], t[2], t[3]) for t in connection.execute(visit_times).fetchall()]
   
    frequencies = [0 for i in xrange(0,12)]
    for start, end, t_start, t_end in visit_times:
        start_hour = datetime.datetime.fromtimestamp(start + t_start).hour
        end_hour = datetime.datetime.fromtimestamp(end + t_end).hour 
        for i in xrange(start_hour, end_hour + 1):
            frequencies[i/2] += 1 
    return frequencies

def intersect_len(a, b):
    return len(list(set(a) & set(b))) * 1.0

def union_len(a, b):
     return len(list(set(a) | set(b))) * 1.0


def bluetooth_frequency_and_variety(place, user):
    btrelation = get_table("btrelation", metadata)
    visit_times = select([visits_10min.c.time_start, visits_10min.c.time_end]).where(and_(visits_10min.c.userid == user, visits_10min.c.placeid == place)) 
    visit_times = [(t[0], t[1]) for t in connection.execute(visit_times).fetchall()]
    average_freq = 0.0
    applications = set()
    bluetooth_visits = []
    bluetooth_count = 0.0
    user_records = [(r[0], r[1]) for r in connection.execute(select([records.c.db_key, records.c.time]).where(and_(records.c.userid == user,records.c.type == 'bt'))).fetchall()]
    for start, end in visit_times:
        filtered_keys = [r[0] for r in user_records if r[1] >= start and r[1] <=end]
        if len(filtered_keys) == 0:
            continue
        distinct_bluetooths = [r[0] for r in connection.execute(select([btrelation.c.id_bnetworks]).where(btrelation.c.db_key.in_(filtered_keys)).distinct()).fetchall()]
        if len(distinct_bluetooths) == 0:
            continue
        bluetooth_count += len(distinct_bluetooths)
        bluetooth_visits.append(distinct_bluetooths)
    if len(bluetooth_visits) < 2:
        variation = 0
    else:
        import itertools
        count = 0.0
        variation = 0.0
        for a, b in itertools.permutations(bluetooth_visits, 2):
            variation += intersect_len(a, b)/union_len(a,b)
            count += 1
        variation /= count    

    return bluetooth_count/len(visit_times), variation

def wlan_frequency_and_variety(place, user):
    wlanrelation = get_table("wlanrelation", metadata)
    visit_times = select([visits_10min.c.time_start, visits_10min.c.time_end]).where(and_(visits_10min.c.userid == user, visits_10min.c.placeid == place)) 
    visit_times = [(t[0], t[1]) for t in connection.execute(visit_times).fetchall()]
    average_freq = 0.0
    applications = set()
    bluetooth_visits = []
    bluetooth_count = 0.0
    user_records = [(r[0], r[1]) for r in connection.execute(select([records.c.db_key, records.c.time]).where(and_(records.c.userid == user,records.c.type == 'wlan'))).fetchall()]
    for start, end in visit_times:
        filtered_keys = [r[0] for r in user_records if r[1] >= start and r[1] <=end]
        if len(filtered_keys) == 0:
            continue
        distinct_bluetooths = [r[0] for r in connection.execute(select([wlanrelation.c.id_wnetworks]).where(wlanrelation.c.db_key.in_(filtered_keys)).distinct()).fetchall()]
        if len(distinct_bluetooths) == 0:
            continue
        bluetooth_count += len(distinct_bluetooths)
        bluetooth_visits.append(distinct_bluetooths)
    if len(bluetooth_visits) < 2:
        variation = 0
    else:
        import itertools
        count = 0.0
        variation = 0.0
        for a, b in itertools.permutations(bluetooth_visits, 2):
            variation += intersect_len(a, b)/union_len(a,b)
            count += 1
        variation /= count    

    return bluetooth_count/len(visit_times), variation


def charging_and_silent_frequency(place, user):
    sys = get_table("sys", metadata)
    visit_times = select([visits_10min.c.time_start, visits_10min.c.time_end]).where(and_(visits_10min.c.userid == user, visits_10min.c.placeid == place)) 
    visit_times = [(t[0], t[1]) for t in connection.execute(visit_times).fetchall()]
    charging = silent = 0.0
    user_records = [(r[0], r[1]) for r in connection.execute(select([records.c.db_key, records.c.time]).where(and_(records.c.userid == user,records.c.type == 'sys'))).fetchall()]
    if len(user_records) == 0:
        return 0, 0
    num_visits = len(visit_times)
    for start, end in visit_times:
        filtered = [r[0] for r in user_records if r[1] >= start and r[1] <=end]
        if(len(filtered)) == 0:
            continue
        l = len(filtered) * 1.0
        charging += (connection.execute(select([func.count(sys.c.db_key)]).where(and_(sys.c.charging != 0, sys.c.db_key.in_(filtered)))).fetchall()[0][0])/l
        silent += (connection.execute(select([func.count(sys.c.db_key)]).where(and_(sys.c.ring == 'silent', sys.c.db_key.in_(filtered)))).fetchall()[0][0])/l
    return (charging/num_visits, silent/num_visits)



def extract_features(place_id, user_id):
    features = []
    charging, silent = charging_and_silent_frequency(place_id, user_id)
    features.append(charging)
    features.append(silent)
    wlanf, wlanv = wlan_frequency_and_variety(place_id, user_id)
    btf, btv = bluetooth_frequency_and_variety(place_id, user_id)
    features.append(wlanf)
    features.append(wlanv)
    features.append(btf)
    features.append(btv)
    frequencies = time_features(place_id, user_id)
    features.extend(frequencies)
    h_w_freq = holiday_to_weekday(place_id, user_id)
    features.append(h_w_freq)
    call_out, missed_call, textout = communication_frequency(place_id, user_id)
    ast = average_stay_time(place_id, user_id)
    rel_freq = relative_frequency(place_id, user_id)
    dist = distance_from_most_visited_place(place_id, user_id)
    mu_freq = media_usage_frequency(place_id, user_id)
    calendar_frequency = calendar_time_frequency(place_id, user_id)
    app_freq, app_div = application_frequency_and_variety(place_id, user_id)
    process_freq, process_type = process_usage_frequency(place_id, user_id)
    features.append(rel_freq)
    features.append(dist)
    features.append(calendar_frequency)
    features.append(app_freq)
    features.append(app_div)
    features.append(process_freq)
    features.append(process_type)
    features.append(mu_freq)
    features.append(ast)
    features.extend([call_out, missed_call, textout])
    return features

def write_features_to_csv(filename, places_features):
    with open(filename, "w") as f:
        writer = csv.writer(f)
        for place, user in places_features.keys():
            row = []
            row.append(place)
            row.append(user)
            label, features = places_features[(place, user)]
            row.append(label)
            row.extend(features)
            writer.writerow(row)

if __name__ == "__main__":
    query = select([places_location.c.placeid, places_location.c.userid, places_location.c.place_label_int])
    places_with_label = [(r[0], r[1], r[2]) for r in connection.execute(query).fetchall()]
    places_features = {(place,user): (label, extract_features(place, user)) for (place,user,label) in places_with_label}
    write_features_to_csv("features_general_new.csv", places_features)
