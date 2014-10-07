import csv
import math

import foursquare
from sqlalchemy.sql import select

from spams.db.utils import create_postgres_engine, get_table, get_metadata, get_connection
from spams.config import FOURSQUARE_CLIENT_ID, FOURSQUARE_CLIENT_SECRET


def h_distance(origin, destination):
    lat1, lon1 = origin
    lat2, lon2 = destination
    radius = 6371  # km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) * math.sin(dlat / 2) + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) * math.sin(dlon / 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = radius * c
    return d * 1000


if __name__ == "__main__":
    client = foursquare.Foursquare(client_id=FOURSQUARE_CLIENT_ID, client_secret=FOURSQUARE_CLIENT_SECRET)
    engine = create_postgres_engine()
    metadata = get_metadata(engine)
    connection = get_connection(engine)
    output_file = open("nearby_places.csv", "w")
    writer = csv.writer(output_file)

    places_location = get_table("places_location", metadata)
    query = select([places_location.c.latitude, places_location.c.longitude, places_location.c.userid, places_location.c.placeid, places_location.c.place_label])
    results = connection.execute(query).fetchall()
    l = len(results)
    for i in range(0, l - 1):
        for j in range(i + 1, l):
            origin = results[i]
            dest = results[j]
            t1 = (origin[2], origin[3])
            if len(str(origin[0]).split(".")[-1]) < 3 and len(str(origin[1]).split(".")[-1]) < 3 and len(str(dest[0]).split(".")[-1]) < 3 and len(str(dest[1]).split(".")[-1]) < 3:
                continue
            t2 = (dest[2], dest[3])
            d = h_distance((origin[0], origin[1]), (dest[0], dest[1]))
            if d < 20:
                print str(t1[0]) + "," + str(t1[1]) + "\t" + str(t2[0]) + "," + str(t2[1]) + "\t" + str(d) + "\t" + origin[4] + "\t" + dest[4] + "\t" + str(origin[0]) + "\t" + str(origin[1]) + "\t" + str(dest[0]) + "\t" + str(dest[1])
