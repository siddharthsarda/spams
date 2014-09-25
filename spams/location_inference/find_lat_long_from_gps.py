import csv

from sqlalchemy.sql import select, and_

from spams.db.utils import create_postgres_engine, get_table, get_metadata, get_connection


if __name__ == "__main__":
    engine = create_postgres_engine()
    metadata = get_metadata(engine)
    connection = get_connection(engine)
    records = get_table("records", metadata)
    gps = get_table("gps", metadata)
    ten_min_visits = get_table("visits_10min", metadata)
    twenty_min_visits = get_table("visits_20min", metadata)
    all_places_r = []
    with open("output", "r") as f:
        reader = csv.reader(f, delimiter=" ")
        for r in reader:
            all_places_r.append((r[0], r[1]))
    lat_long = {}
    print "Starting to iterate over all places"
    no_wifi_visits_10 = {}
    no_wifi_visits_20 = {}
    print len(all_places_r)
    count = 0
    for userid, placeid in all_places_r:
        lat_long[(userid, placeid)] = []
        no_wifi_visits_10[(userid, placeid)] = []
        no_wifi_visits_20[(userid, placeid)] = []
        ten_places = select([ten_min_visits.c.time_start, ten_min_visits.c.time_end], and_(ten_min_visits.c.userid == userid, ten_min_visits.c.placeid == placeid)).order_by(ten_min_visits.c.time_start).limit(50)
        twenty_places = select([twenty_min_visits.c.time_start, twenty_min_visits.c.time_end], and_(twenty_min_visits.c.userid == userid, twenty_min_visits.c.placeid == placeid)).order_by(twenty_min_visits.c.time_start).limit(50)
        records_gps_type = select([records], and_(records.c.type == "gps", records.c.userid == userid)).alias("records_g")
        joined = records_gps_type.join(gps, onclause=gps.c.db_key == records_gps_type.c.db_key)
        for time_start, time_end in connection.execute(ten_places):
            locs = select([joined.c.gps_latitude, joined.c.gps_longitude]).where(gps.c.db_key == records_gps_type.c.db_key).where(and_(joined.c.gps_time >= time_start, joined.c.gps_time <= time_end))
            locs = locs.distinct()
            locs = connection.execute(locs).fetchall()
            lat_long[(userid, placeid)].extend(locs)
        for time_start, time_end in connection.execute(twenty_places):
            locs = select([joined.c.gps_latitude, joined.c.gps_longitude]).where(gps.c.db_key == records_gps_type.c.db_key).where(and_(joined.c.gps_time >= time_start, joined.c.gps_time <= time_end))
            locs = locs.distinct()
            locs = connection.execute(locs).fetchall()
            lat_long[(userid, placeid)].extend(locs)
        count += 1
        if ((count % 20) == 0):
            print count
    with open("20_gps_files_from_gps.csv", "w") as f:
        writer = csv.writer(f)
        for userid, placeid in lat_long.keys():
            to_write = [(str(a[0]), str(a[1])) for a in lat_long[(userid, placeid)]]
            to_write = list(set(to_write))
            writer.writerow([userid, placeid, to_write])
