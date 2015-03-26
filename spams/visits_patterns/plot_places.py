import os
os.environ['MPLCONFIGDIR'] = "/local/.config/matplotlib"
print(os.environ.get('MPLCONFIGDIR'))

from datetime import datetime
import textwrap

from sqlalchemy import and_, func
from sqlalchemy.sql import select
import matplotlib.pyplot as plt

from spams.db.utils import setup_database, get_table
from spams.utils import draw_barplot
from spams.mappings import LABEL_PLACE_MAPPING, AGE_MAPPING, GENDER_MAPPING, WORKING_MAPPING, BILL_MAPPING
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
    tables = ["visits_10min"]
    for table in tables:
        consolidated = return_joined_table(table, metadata)
        for place_label in xrange(1, 11):
            query = select([consolidated.c["visits_joined_" + table + "_time_start"]], consolidated.c["visits_joined_places_place_label"] == place_label)
            start_times = connection.execute(query).fetchall()
            hours = [0 for i in xrange(24)]
            for start_time in start_times:
                hours[datetime.fromtimestamp(start_time[0]).hour] += 1
            place_name = LABEL_PLACE_MAPPING[place_label]
            filename = place_name.replace(';', '_').replace(" ", "_") + "_hours.png"
            draw_barplot(hours, x_ticks=xrange(24), xlabel="Hour of Day", ylabel="Number of Checkins", title="%s Visits by Hours" % (place_name), save_as=os.path.join("/local", "thesis", "plots", filename))


def plot_start_time_day():
    day_dict = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    metadata, connection = setup_database()
    tables = ["visits_10min"]
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
            filename = place_name.replace(';', '_').replace(" ", "_") + "_day.png"
            draw_barplot(days, x_ticks=day_dict, xlabel="Day of week", ylabel="Number of Checkins", title="%s Visits by Days" % (place_name), save_as=os.path.join("/local", "thesis", "plots", filename))


def plot_gender():
    metadata, connection = setup_database()
    tables = ["visits_10min"]
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
        ax.set_ylabel("Count", fontsize = 24, fontweight = 'bold')
        
        ax.set_xlabel("Place Category", fontsize=24, fontweight = 'bold')
        ax.set_title("Visits Across Gender", fontsize=32, fontweight='bold')
        xticks_values = [LABEL_PLACE_MAPPING[i] for i in xrange(1, 11)]
        xticks_values = [textwrap.fill(text,10) for text in xticks_values]
        ax.set_xticks([i + width for i in xrange(1, 11)])
        ax.set_xticklabels(xticks_values)
        #autolabel(rects1, gender_checkins[0])
        #autolabel(rects2, gender_checkins[1])
        plt.show()


def plot_age_groups():
    metadata, connection = setup_database()
    tables = ["visits_10min"]
    for table in tables:
        consolidated = return_joined_table(table, metadata)

        for place_label in xrange(1, 11):
            age_checkins = []
            for age_group in xrange(1, 9):
                query = select([func.count()], and_(consolidated.c["visits_joined_places_place_label"] == place_label, consolidated.c["demographics_age_group"] == age_group))
                result = connection.execute(query).fetchall()
                age_checkins.append(result[0][0])
            fig, ax = plt.subplots()
            rects = ax.bar(xrange(1, 9), age_checkins)
            ax.set_ylabel("Count", fontsize=30, fontweight='bold')
            ax.set_xlabel("Age groups", fontsize=30, fontweight='bold')
            ax.set_title(LABEL_PLACE_MAPPING[place_label] + " Visits across Age Groups", fontsize=36, fontweight='bold')
            xticks_values = [AGE_MAPPING[i] for i in xrange(1, 9)]
            ax.set_xticks([i + 0.35 for i in xrange(1, 9)])
            ax.set_xticklabels(xticks_values)
            #autolabel(rects, age_checkins)
            place_name = LABEL_PLACE_MAPPING[place_label]
            filename = place_name.replace(";", "_").replace(" ", "_")  +  "_" + "age.png"
            fig.set_size_inches((15, 12))
            fig.savefig(filename, dpi=100)
            plt.close(fig)


def plot_working_groups():
    metadata, connection = setup_database()
    tables = ["visits_10min"]
    for table in tables:
        consolidated = return_joined_table(table, metadata)
        for place_label in xrange(1, 11):
            working_checkins = []
            for working_group in xrange(1, 9):
                query = select([func.count()], and_(consolidated.c["visits_joined_places_place_label"] == place_label, consolidated.c["demographics_working"] == working_group))
                result = connection.execute(query).fetchall()
                working_checkins.append(result[0][0])
            #fig, ax = plt.subplots()
            #ax.legend((xrange(1,9)), xrange(1, 9))
            #rects = ax.bar(xrange(1, 9), working_checkins)
            #ax.set_ylabel("Count", fontsize=30, fontweight='bold')
            #ax.set_xlabel("Working groups", fontsize=30, fontweight='bold')
            #ax.set_title(LABEL_PLACE_MAPPING[place_label] + " Visits across Work Groups", fontsize=36, fontweight='bold')
            x_ticks = [WORKING_MAPPING[i] for i in xrange(1, 9)]
            #xticks_values = [textwrap.fill(text,7) for text in xticks_values]

            #ax.set_xticks([i + 0.3 for i in xrange(1, 9)])
            #ax.set_xticklabels(xticks_values)
            #autolabel(rects, working_checkins)
            place_name = LABEL_PLACE_MAPPING[place_label]
            filename = place_name.replace(";", "_").replace(" ", "_") + "_" + "workgroup.png"
            #fig.set_size_inches((15, 12))
            #fig.savefig(filename, dpi=100)
            #plt.close(fig)
            draw_barplot(working_checkins, x_ticks=[textwrap.fill(text,10) for text in x_ticks], xlabel="Working Status", ylabel="Visits", title=LABEL_PLACE_MAPPING[place_label] + " Visits across Employment Status", save_as=os.path.join("/local", "thesis", "plots", "working",filename), width=0.35)


def plot_demographics():
    metadata, connection = setup_database()
    demographics = get_table("demographics", metadata)

    gender_query = select([demographics.c.gender, func.count(demographics.c.gender)]).group_by(demographics.c.gender)
    result = connection.execute(gender_query).fetchall()
    result = [r for r in result if r[0] is not None]
    result = sorted(result, key=lambda x: x[0])
    vals = [r[1] for r in result]
    x_ticks = [GENDER_MAPPING[r[0]] for r in result]
    filename = "gender.png"
    draw_barplot(vals, x_ticks=x_ticks, xlabel="Gender", ylabel="Count", title="Gender Distribution", save_as=os.path.join("/local", "thesis", "plots", filename), width=0.35)

    age_query = select([demographics.c.age_group, func.count(demographics.c.age_group)]).group_by(demographics.c.age_group)
    result = connection.execute(age_query).fetchall()
    result = [r for r in result if r[0] is not None]
    result = sorted(result, key=lambda x: x[0])
    vals = [r[1] for r in result]
    x_ticks = [AGE_MAPPING[r[0]] for r in result]
    filename = "age.png"
    draw_barplot(vals, x_ticks=x_ticks, xlabel="Age Group", ylabel="Count", title="Age Distribution", save_as=os.path.join("/local", "thesis", "plots", filename), width=0.35)

    working_query = select([demographics.c.working, func.count(demographics.c.working)]).group_by(demographics.c.working)
    result = connection.execute(working_query).fetchall()
    print result
    result = [r for r in result if r[0] is not None]
    result = sorted(result, key=lambda x: x[0])
    vals = [r[1] for r in result]
    x_ticks = [WORKING_MAPPING[r[0]] for r in result]
    filename = "working.png"
    draw_barplot(vals, x_ticks=[textwrap.fill(text,10) for text in x_ticks], xlabel="Employment Status", ylabel="Count", title="Employment Status Distribution", save_as=os.path.join("/local", "thesis", "plots", filename), width=0.35)

    bill_query = select([demographics.c.phone_bill, func.count(demographics.c.phone_bill)]).group_by(demographics.c.phone_bill)
    result = connection.execute(bill_query).fetchall()
    result = [r for r in result if r[0] is not None]
    result = sorted(result, key=lambda x: x[0])
    vals = [r[1] for r in result]
    x_ticks = [BILL_MAPPING[r[0]] for r in result]
    filename = "bill.png"
    draw_barplot(vals, x_ticks=x_ticks, xlabel="Bill", ylabel="Count", title="Bill Distribution", save_as=os.path.join("/local", "thesis", "plots", filename), width=0.35)

    bill_query = select([demographics.c.nb_12, demographics.c.nb_12_18, demographics.c.nb_18_30, demographics.c.nb_30_40, demographics.c.nb_40_50, demographics.c.nb_50_65, demographics.c.nb_65])
    result = connection.execute(bill_query).fetchall()
    result = [sum([a for a in r if a is not None]) for r in result if r is not None]
    s = set(result)
    print s
    vals = []
    x_ticks = []
    for elem in s:
        if elem > 13:
            continue
        x_ticks.append(elem)
        vals.append(result.count(elem))
    #vals = [r[1] for r in result]
    #x_ticks = [BILL_MAPPING[r[0]] for r in result]
    filename = "family.png"
    draw_barplot(vals, x_ticks=x_ticks, xlabel="Number of members in family", ylabel="Count", title="Number of Family Members Distribution", save_as=os.path.join("/local", "thesis", "plots", filename), width=0.35)



if __name__ == "__main__":
    import matplotlib
    font = {'family' : 'normal',
            'weight' : 'bold',
            'size'   : 18}
    matplotlib.rc('font', **font)
    plot_working_groups()
