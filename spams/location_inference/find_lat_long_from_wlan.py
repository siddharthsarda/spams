import csv

from sqlalchemy.sql import select, and_

from spams.db.utils import create_postgres_engine, get_table, get_metadata, get_connection


if __name__ == "__main__":
    engine = create_postgres_engine()
    metadata = get_metadata(engine)
    connection = get_connection(engine)
    records = get_table("records", metadata)
    gpswlan = get_table("gpswlan", metadata)
    ten_min_visits = get_table("visits_10min", metadata)
    twenty_min_visits = get_table("visits_20min", metadata)
    places = get_table("places", metadata)
    all_places = select([places.c.userid, places.c.placeid]).distinct()
    print all_places
    all_places_r = connection.execute(all_places)
    lat_long = {}
    print "Starting to iterate over all places"
    no_wifi_visits_10 = {}
    no_wifi_visits_20 = {}
    print all_places_r.rowcount
    count = 0
    for userid, placeid in all_places_r:
        lat_long[(userid, placeid)] = []
        no_wifi_visits_10[(userid, placeid)] = []
        no_wifi_visits_20[(userid, placeid)] = []
        ten_places = select([ten_min_visits.c.time_start, ten_min_visits.c.time_end], and_(ten_min_visits.c.userid == userid, ten_min_visits.c.placeid == placeid)).limit(5)
        twenty_places = select([twenty_min_visits.c.time_start, twenty_min_visits.c.time_end], and_(twenty_min_visits.c.userid == userid, twenty_min_visits.c.placeid == placeid)).limit(5)
        records_gpswlan_type = select([records], and_(records.c.type == "gpswlan", records.c.userid == userid)).alias("records_g")
        joined = records_gpswlan_type.join(gpswlan, onclause=gpswlan.c.db_key == records_gpswlan_type.c.db_key)
        for time_start, time_end in connection.execute(ten_places):
            wlan_locs = select([joined.c.gpswlan_latitude, joined.c.gpswlan_longitude]).where(gpswlan.c.db_key == records_gpswlan_type.c.db_key).where(and_(joined.c.records_g_time >= time_start, joined.c.records_g_time <= time_end))
            wlan_locs = wlan_locs.distinct()
            wlan_locs = connection.execute(wlan_locs).fetchall()
            if (len(wlan_locs) == 0):
                no_wifi_visits_10[(userid, placeid)].append((time_start, time_end))
            lat_long[(userid, placeid)].extend(wlan_locs)
        for time_start, time_end in connection.execute(twenty_places):
            wlan_locs = select([joined.c.gpswlan_latitude, joined.c.gpswlan_longitude]).where(gpswlan.c.db_key == records_gpswlan_type.c.db_key).where(and_(joined.c.records_g_time >= time_start, joined.c.records_g_time <= time_end))
            wlan_locs = wlan_locs.distinct()
            wlan_locs = connection.execute(wlan_locs).fetchall()
            if (len(wlan_locs) == 0):
                no_wifi_visits_20[(userid, placeid)].append((time_start, time_end))
            lat_long[(userid, placeid)].extend(wlan_locs)
        count += 1
        if ((count % 50) == 0):
            print count
    with open("gps_files_from_wifi.csv", "w") as f:
        writer = csv.writer(f)
        for userid, placeid in lat_long.keys():
            to_write = [(str(a[0]), str(a[1])) for a in lat_long[(userid, placeid)]]
            to_write = list(set(to_write))
            writer.writerow([userid, placeid, to_write])
    with open("no_wifi_10.csv", "w") as f:
        writer = csv.writer(f)
        for userid, placeid in no_wifi_visits_10.keys():
            if len(no_wifi_visits_10[(userid, placeid)]) == 0:
                continue
            to_write = [(str(a[0]), str(a[1])) for a in no_wifi_visits_10[(userid, placeid)]]
            to_write = list(set(to_write))
            writer.writerow([userid, placeid, to_write])
    with open("no_wifi_20.csv", "w") as f:
        writer = csv.writer(f)
        for userid, placeid in no_wifi_visits_20.keys():
            if len(no_wifi_visits_20[(userid, placeid)]) == 0:
                continue
            to_write = [(str(a[0]), str(a[1])) for a in no_wifi_visits_20[(userid, placeid)]]
            to_write = list(set(to_write))
            writer.writerow([userid, placeid, to_write])
