import csv

from sqlalchemy.sql import select

from spams.db.utils import create_postgres_engine, get_connection, get_metadata, get_table

PLACES = "places.csv"
MAPPING = "mapping"
OUTPUT = "places_with_mapping.csv"


if __name__ == "__main__":
    places_file = open(PLACES, "r")
    mapping_file = open(MAPPING, "r")
    output_file = open(OUTPUT, "w")
    places_reader = csv.reader(places_file)
    mapping_reader = csv.reader(mapping_file)
    mapping = {}
    for row in mapping_reader:
        mapping[int(row[0])] = row[1]
    output_writer = csv.writer(output_file)
    engine = create_postgres_engine()
    connection = get_connection(engine)
    metadata = get_metadata(engine)
    places = get_table("places", metadata)
    labels = connection.execute(select([places.c.userid, places.c.placeid, places.c.place_label])).fetchall()
    places_dict = {}
    for a in labels:
        places_dict[(a[0], a[1])] = a[2]
    for row in places_reader:
        label_int = places_dict[(int(row[0]), int(row[1]))]
        row.append(mapping[label_int])
        row.append(label_int)
        output_writer.writerow(row)
    places_file.close()
    mapping_file.close()
    output_file.close()
