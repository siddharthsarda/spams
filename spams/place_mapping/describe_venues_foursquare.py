import csv
import operator
from collections import defaultdict

import foursquare
from sqlalchemy.sql import select

from spams.db.utils import create_postgres_engine, get_table, get_metadata, get_connection
from spams.config import FOURSQUARE_CLIENT_ID, FOURSQUARE_CLIENT_SECRET


def safe_str(s):
    return s.encode("ascii", "ignore")


def build_category_hierarchy_dict(client):
    category_hierarchy = {}
    categories = client.venues.categories()['categories']
    categories_stack = []
    for category in categories:
        categories_stack.append(category)
        category_hierarchy[safe_str(category['name'])] = None
    while len(categories_stack) > 0:
        parent_category = categories_stack.pop()
        if 'categories' in parent_category:
            for c in parent_category['categories']:
                category_hierarchy[safe_str(c['name'])] = safe_str(parent_category['name'])
                categories_stack.append(c)
    return category_hierarchy


def find_top_level_category(category_name, category_hierarchy_dict):
    top_level = category_name
    while category_hierarchy_dict[top_level] is not None:
        top_level = category_hierarchy_dict[top_level]

    return top_level


if __name__ == "__main__":
    client = foursquare.Foursquare(client_id=FOURSQUARE_CLIENT_ID, client_secret=FOURSQUARE_CLIENT_SECRET)
    category_hierarchy = build_category_hierarchy_dict(client)
    engine = create_postgres_engine()
    metadata = get_metadata(engine)
    connection = get_connection(engine)
    output_file = open("nearby_places.csv", "w")
    writer = csv.writer(output_file)

    places_location = get_table("places_location", metadata)
    query = select([places_location.c.latitude, places_location.c.longitude, places_location.c.userid, places_location.c.placeid, places_location.c.place_label])
    results = connection.execute(query)
    for r in results:
        ll = str(r[0]) + "," + str(r[1])
        venues = client.venues.search(params={'ll': ll, 'radius': 50, 'intent': 'browse'})
        to_write = []
        d = defaultdict(int)
        for venue in venues["venues"]:
            name = venue['name']
            name = name.encode('ascii', 'ignore')
            distance = venue['location']['distance']
            categories = []
            for category in venue['categories']:
                cname = safe_str(category['name'])
                d[find_top_level_category(cname, category_hierarchy)] += 1
        if len(d) == 0:
            writer.writerow([str(r[2]) + "," + str(r[3]), r[4], "No category", r[0], r[1]])
            continue
        sum = reduce(lambda x, y: x + y, d.values())
        foursq_categories = []
        for key in d:
            d[key] = (d[key] * 1.0) / sum
        sorted_d = sorted(d.items(), key=operator.itemgetter(1))
        for key, value in reversed(sorted_d):
            foursq_categories.append(str(key) + ":" + str(value))
        foursq_categories = ";".join(foursq_categories)

        to_write.append([str(r[2]) + "," + str(r[3]), r[4], foursq_categories, r[0], r[1]])
        to_write = sorted(to_write, key=lambda t: t[3])
        for row in to_write:
            writer.writerow(row)
    output_file.close()
