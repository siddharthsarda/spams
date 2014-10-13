import csv
import math
from collections import defaultdict
import operator

from googleplaces import GooglePlaces
from sqlalchemy.sql import select

from spams.db.utils import create_postgres_engine, get_table, get_metadata, get_connection
from spams.config import GOOGLE_API_KEY


FORBIDDEN_TYPES = ["point_of_interest", "route", "locality", "political", "establishment", "sublocality_level_2", "sublocality", "sublocality_level_1", "natural_feature"]


def get_foursquare_mapping():
    category_mapping_dict = {}
    with open("category_mapping.csv", "r") as f:
        reader = csv.reader(f)
        for row in reader:
            category_mapping_dict[row[0]] = row[1]
    return category_mapping_dict


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
    foursquare_categories = get_foursquare_mapping()

    places_location = get_table("places_location", metadata)
    query = select([places_location.c.latitude, places_location.c.longitude, places_location.c.userid, places_location.c.placeid, places_location.c.place_label])
    results = connection.execute(query)
    for r in results:
        venues = client.nearby_search(lat_lng={'lat': r[0], 'lng': r[1]}, radius=50)
        to_write = []
        categories = defaultdict(int)
        if len(venues.places) == 0:
            writer.writerow([str(r[2]) + "," + str(r[3]), r[4], "No Category", r[0], r[1]])
            continue
        for venue in venues.places:
            venue.get_details()
            name = venue.name
            name = name.encode('ascii', 'ignore')
            # distance = h_distance((float(r[0]), float(r[1])), (venue.geo_location['lat'], venue.geo_location['lng']))
            types = [type for type in venue.types if type not in FORBIDDEN_TYPES]
            for type in types:
                categories[foursquare_categories[type]] += 1
        if len(categories) == 0:
            writer.writerow([str(r[2]) + "," + str(r[3]), r[4], "No Category", r[0], r[1]])
            continue

        sum = reduce(lambda x, y: x + y, categories.values())
        google_categories = []
        for key in categories:
            categories[key] = (categories[key] * 1.0) / sum
        categories_sorted = sorted(categories.items(), key=operator.itemgetter(1))
        for key, value in reversed(categories_sorted):
            google_categories.append(str(key) + ":" + str(value))
        writer.writerow([str(r[2]) + "," + str(r[3]), r[4], ";".join(google_categories), r[0], r[1]])
    output_file.close()
