import csv
import math

from googleplaces import GooglePlaces
from sqlalchemy.sql import select

from spams.db.utils import create_postgres_engine, get_table, get_metadata, get_connection
from spams.config import GOOGLE_API_KEY


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
    client = GooglePlaces(GOOGLE_API_KEY)
    engine = create_postgres_engine()
    metadata = get_metadata(engine)
    connection = get_connection(engine)
    output_file = open("nearby_places_google.csv", "w")
    writer = csv.writer(output_file)

    places_location = get_table("places_location", metadata)
    query = select([places_location.c.latitude, places_location.c.longitude, places_location.c.userid, places_location.c.placeid, places_location.c.place_label])
    results = connection.execute(query)
    for r in results:
        venues = client.nearby_search(lat_lng={'lat': r[0], 'lng': r[1]}, radius=50)
        to_write = []
        for venue in venues.places:
            venue.get_details()
            name = venue.name
            name = name.encode('ascii', 'ignore')
            distance = h_distance((float(r[0]), float(r[1])), (venue.geo_location['lat'], venue.geo_location['lng']))
            categories = ";".join(venue.types)
            to_write.append([str(r[2]) + "," + str(r[3]), r[4], name, distance, categories, r[0], r[1]])
        to_write = sorted(to_write, key=lambda t: t[3])
        for row in to_write:
            writer.writerow(row)
    output_file.close()
