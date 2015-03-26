import csv

import foursquare
from sqlalchemy.sql import select

from spams.db.utils import create_postgres_engine, get_table, get_metadata, get_connection
from spams.config import FOURSQUARE_CLIENT_ID, FOURSQUARE_CLIENT_SECRET


if __name__ == "__main__":
    client = foursquare.Foursquare(client_id=FOURSQUARE_CLIENT_ID, client_secret=FOURSQUARE_CLIENT_SECRET)
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
        for venue in venues["venues"]:
            name = venue['name']
            name = name.encode('ascii', 'ignore')
            distance = venue['location']['distance']
            categories = []
            for category in venue['categories']:
                cname = category['name']
                cname = cname.encode('ascii', 'ignore')
                categories.append(cname)
                categories = ";".join(categories)
                to_write.append([str(r[2]) + "," + str(r[3]), r[4], name, distance, categories, r[0], r[1]])
        to_write = sorted(to_write, key=lambda t: t[3])
        for row in to_write:
            writer.writerow(row)
        #break    
    output_file.close()
