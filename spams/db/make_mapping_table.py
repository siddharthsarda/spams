import csv

from sqlalchemy import Table, Column, Integer, String
from sqlalchemy.types import Numeric

from spams.db.utils import create_postgres_engine, get_metadata, get_connection, get_table


FILENAME = "places_with_mapping.csv"


if __name__ == "__main__":
    engine = create_postgres_engine()
    metadata = get_metadata(engine)
    connection = get_connection(engine)
    table_name = 'places_location'
    if not engine.dialect.has_table(connection, table_name):
        places_location_mapping = Table(table_name, metadata,
                                        Column('id', primary_key = True)
                                        Column('userid', Integer),
                                        Column('placeid', Integer),
                                        Column('latitude', Numeric),
                                        Column('latitude_std', Numeric),
                                        Column('longitude', Numeric),
                                        Column('longitude_std', Numeric),
                                        Column('method', String),
                                        Column('place_label', String),
                                        Column('place_label_int', Integer))
        metadata.create_all(engine)
        print "Created Table"
    else:
        print "Table exists"
    places_location_mapping = get_table(table_name, metadata)
    with open(FILENAME) as f:
        reader = csv.reader(f)
        for row in reader:
            ins = places_location_mapping.insert().values(tuple(row))
            print connection.execute(ins)
