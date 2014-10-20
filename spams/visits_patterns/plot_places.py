import os
from datetime import datetime

from sqlalchemy import and_, func
from sqlalchemy.sql import select
import matplotlib.pyplot as plt

from spams.db.utils import setup_database, get_table
from spams.utils import draw_barplot
from spams.location_inference.label_place_mapping import LABEL_PLACE_MAPPING
from spams.utils import autolabel


def return_joined_table(tablename, metadata):
    visits = get_table(tablename, metadata)
    places = get_table("places", metadata)
    demographics = get_table("demographics", metadata)
    visits_with_places = places.join(visits, onclause=and_(places.c.userid == visits.c.userid, places.c.placeid == visits.c.placeid)).alias("visits_joined")
    visits_with_places_and_demographics = visits_with_places.join(demographics, visits_with_places.c[tablename + "_userid"] == demographics.c.userid).alias("consolidated")
    return visits_with_places_and_demographics


def plot_start_time_hour():
    metadata, connection = setup_database()
    tables = ["visits_10min", "visits_20min"]
    for table in tables:
        consolidated = return_joined_table(table, metadata)
        for place_label in xrange(1, 11):
            query = select([consolidated.c["visits_joined_" + table + "_time_start"]], consolidated.c["visits_joined_places_place_label"] == place_label)
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
        consolidated = return_joined_table(table, metadata)
        for place_label in xrange(1, 11):
            query = select([consolidated.c["visits_joined_" + table + "_time_start"]], consolidated.c["visits_joined_places_place_label"] == place_label)
            start_times = connection.execute(query).fetchall()
            days = [0 for i in xrange(7)]
            for start_time in start_times:
                current_day = datetime.fromtimestamp(start_time[0]).weekday()
                days[current_day] += 1
            place_name = LABEL_PLACE_MAPPING[place_label]
            filename = place_name + "_" + table + "_day.png"
            draw_barplot(days, x_ticks=day_dict, xlabel="Day of week", ylabel="Number of Checkins", title="%s start times in days for table %s" % (place_name, table), save_as=os.path.join("/local", "thesis", "plots", filename))


def plot_gender():
    metadata, connection = setup_database()
    tables = ["visits_10min", "visits_20min"]
    for table in tables:
        consolidated = return_joined_table(table, metadata)
        print connection.execute(select([func.count()], consolidated.c["demographics_gender"] == 2)).fetchall()
        gender_checkins = []
        for gender in (0, 1):
            gender_checkins.append([])
            for place_label in xrange(1, 11):
                query = select([func.count()], and_(consolidated.c["visits_joined_places_place_label"] == place_label, consolidated.c["demographics_gender"] == gender + 1))
                result = connection.execute(query).fetchall()
                gender_checkins[gender].append(result[0][0])
        fig, ax = plt.subplots()
        width = 0.35
        rects1 = ax.bar(xrange(1, 11), gender_checkins[0], width, color='r')
        rects2 = ax.bar([i + width for i in xrange(1, 11)], gender_checkins[1], width, color='g')
        ax.legend((rects1[0], rects2[0]), ('Men', 'Women'))
        ax.set_ylabel("Count")
        xticks_values = [LABEL_PLACE_MAPPING[i] for i in xrange(1, 11)]
        ax.set_xticks([i + width for i in xrange(1, 11)])
        ax.set_xticklabels(xticks_values)
        autolabel(rects1, gender_checkins[0])
        autolabel(rects2, gender_checkins[1])
        plt.show()


AGE_MAPPINGS = {1: "<16", 2: "16-21", 3: "22-27", 4: "28-33", 5: "33-38", 6: "39-44", 7: "45-50", 8: ">50"}


def plot_age_groups():
    metadata, connection = setup_database()
    tables = ["visits_10min", "visits_20min"]
    for table in tables:
        consolidated = return_joined_table(table, metadata)
        age_checkins = []

        for place_label in xrange(1, 11):
            age_checkins = []
            for age_group in xrange(1, 9):
                query = select([func.count()], and_(consolidated.c["visits_joined_places_place_label"] == place_label, consolidated.c["demographics_age_group"] == age_group))
                result = connection.execute(query).fetchall()
                age_checkins.append(result[0][0])
            fig, ax = plt.subplots()
            rects = ax.bar(xrange(1, 9), age_checkins)
            ax.set_ylabel("Count")
            ax.set_xlabel("Age groups")
            ax.set_title(LABEL_PLACE_MAPPING[place_label] + " across age groups for table: " + table)
            xticks_values = [AGE_MAPPINGS[i] for i in xrange(1, 9)]
            ax.set_xticks([i + 0.35 for i in xrange(1, 9)])
            ax.set_xticklabels(xticks_values)
            autolabel(rects, age_checkins)
            place_name = LABEL_PLACE_MAPPING[place_label]
            filename = place_name + "_" + table + "age.png"
            fig.set_size_inches((15, 12))
            fig.savefig(filename, dpi=100)
            plt.close(fig)


if __name__ == "__main__":
    plot_age_groups()
