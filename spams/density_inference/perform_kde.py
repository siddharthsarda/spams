import sys
import numpy as np
from sklearn.neighbors import KernelDensity
from sqlalchemy.sql import select
from sqlalchemy import func
from spams.db.utils import setup_database, get_table
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

if __name__ == "__main__":
    metadata, connection = setup_database()
    places_location = get_table("places_location", metadata)
    min_lat, max_lat, min_long, max_long = connection.execute(select([func.min(places_location.c.latitude), func.max(places_location.c.latitude), func.min(places_location.c.longitude), func.max(places_location.c.longitude)])).fetchall()[0]
    
    for label in xrange(1, 11):
        lat_long_query = select([places_location.c.latitude, places_location.c.longitude]).where(places_location.c.place_label_int==label)
        results = connection.execute(lat_long_query).fetchall()
        latitudes = [float(r[0]) for r in results]
        longitudes = [float(r[1]) for r in results]
        X, Y = np.meshgrid(longitudes, latitudes)
        xy = np.vstack([Y.ravel(), X.ravel()]).T
        #X = np.asarray(longitudes)
        #Y = np.asarray(latitudes)

        #convert to radians
        xy *= np.pi /180.
        kde = KernelDensity(bandwidth=1, metric="haversine", kernel="gaussian", algorithm='ball_tree')
        kde.fit(xy)
        Z = np.exp(kde.score_samples(xy))
        levels = np.linspace(0, Z.max(), 10)
        plt.contourf(X, Y, Z, levels=levels)
        m = Basemap(projection='cyl', llcrnrlat=min_lat - 1, urcrnrlat=max_lat + 1, llcrnrlon=min_long - 1, urcrnrlon=max_long + 1, resolution='c')
        m.drawcountries()
        plt.show()
        break

