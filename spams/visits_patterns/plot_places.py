import os
from datetime import datetime

from sqlalchemy import and_
from sqlalchemy.sql import select

from spams.db.utils import setup_database, get_table
from spams.utils import draw_barplot
from spams.location_inference.label_place_mapping import LABEL_PLACE_MAPPING


def return_joined_table(tablename, metadata):
    visits = get_table(tablename, metadata)
    places = get_table("places", metadata)
    return places.join(visits, onclause=and_(places.c.userid == visits.c.userid, places.c.placeid == visits.c.placeid)).alias("visits_joined")


def plot_start_time_hour():
    metadata, connection = setup_database()
    tables = ["visits_10min", "visits_20min"]
    for table in tables:
        visits_with_places = return_joined_table(table, metadata)
        for place_label in xrange(1, 11):
            query = select([visits_with_places.c[table + "_time_start"]], visits_with_places.c.places_place_label == place_label)
            start_times = connection.execute(query).fetchall()
            hours = [0 for i in xrange(24)]
            for start_time in start_times:
                hours[datetime.fromtimestamp(start_time[0]).hour] += 1
            place_name = LABEL_PLACE_MAPPING[place_label]
            filename = place_name + "_" + table + "_hours.png"
            draw_barplot(hours, x_ticks=xrange(24), xlabel="Hour of Day", ylabel="Number of Checkins", title="%s start times in hours for table %s" % (place_name, table), save_as=os.path.join("/local", "thesis", "plots", filename))


def plot_start_time_day():
    day_dict = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    metadata, connection = setup_database()
    tables = ["visits_10min", "visits_20min"]
    for table in tables:
        visits_with_places = return_joined_table(table, metadata)
        for place_label in xrange(1, 11):
            query = select([visits_with_places.c[table + "_time_start"]], visits_with_places.c.places_place_label == place_label)
            start_times = connection.execute(query).fetchall()
            days = [0 for i in xrange(7)]
            for start_time in start_times:
                current_day = datetime.fromtimestamp(start_time[0]).weekday()
                days[current_day] += 1
            place_name = LABEL_PLACE_MAPPING[place_label]
            filename = place_name + "_" + table + "_day.png"
            draw_barplot(days, x_ticks=day_dict, xlabel="Day of week", ylabel="Number of Checkins", title="%s start times in days for table %s" % (place_name, table), save_as=os.path.join("/local", "thesis", "plots", filename))


if __name__ == "__main__":
    plot_start_time_day()
