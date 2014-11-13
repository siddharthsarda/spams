import sys
import numpy as np
from sklearn.neighbors import KernelDensity
from sqlalchemy.sql import select
from sqlalchemy import func
from spams.db.utils import setup_database, get_table
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from spams.mappings import LABEL_PLACE_MAPPING

if __name__ == "__main__":
    metadata, connection = setup_database()
    places_location = get_table("places_location", metadata)
    min_lat, max_lat, min_long, max_long = connection.execute(select([func.min(places_location.c.latitude), func.max(places_location.c.latitude), func.min(places_location.c.longitude), func.max(places_location.c.longitude)])).fetchall()[0]
    for label in xrange(1, 11):
        print LABEL_PLACE_MAPPING[label] 
        lat_long_query = select([places_location.c.latitude, places_location.c.longitude]).where(places_location.c.place_label_int==label)
        results = connection.execute(lat_long_query).fetchall()
        latitudes = [float(r[0]) for r in results]
        longitudes = [float(r[1]) for r in results]
        X = np.asarray(longitudes)
        Y = np.asarray(latitudes)
        m = Basemap(projection='cyl', llcrnrlat=float(min_lat) - 1, urcrnrlat=float(max_lat) + 1, llcrnrlon=float(min_long) - 1, urcrnrlon=float(max_long) + 1, resolution='c')
        x, y = m(X,Y)
        m.scatter(x,y,3,marker='o',color='k')
        m.drawcountries()
        plt.show()
