import csv
import sys
from decimal import Decimal
import numpy as np


csv.field_size_limit(sys.maxsize)

FILENAME = "backup/gps_files_from_gps.csv"
OUTPUT = "places_from_gps.csv"
if __name__ == "__main__":
    w = open(OUTPUT, "w")
    with open(FILENAME, "r") as f:
        reader = csv.reader(f)
        writer = csv.writer(w)
        for row in reader:
            measurements = eval(row[2])
            measurements = [m for m in measurements if len(m[0].split(".")[-1]) >= 4 and len(m[1].split(".")[-1]) >= 4]
            if len(measurements) == 0:
                continue
            measurements_lat = [Decimal(m[0]) for m in measurements]
            measurements_long = [Decimal(m[1]) for m in measurements]
            if len(measurements) == 1:
                writer.writerow([row[0], row[1], measurements_lat[0], 0, measurements_long[0], 0, "gps"])
                continue
            writer.writerow([row[0], row[1], np.mean(measurements_lat), np.std(measurements_lat), np.mean(measurements_long), np.std(measurements_long), "gps"])
    w.close()
