import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import dateutil.parser
import csv
import random


def remove(l, n):
    for _ in range(int(len(l) * n)):
        l.pop(random.randrange(0, len(l)))
    return l


csv_file = csv.DictReader(open(file='trajectory.csv'))
timestamp_list_s = []
latitude_list_s = []
longitude_list_s = []

for row in csv_file:
    timestamp_list_s.append(row['timestamp'])
    longitude_list_s.append(float(row['longitude']))
    latitude_list_s.append(float(row['latitude']))

timestamp_list = []
for timestamp in timestamp_list_s:
    date = dateutil.parser.parse(timestamp)
    timestamp_list.append(date)

latitude_list = list(map(float, latitude_list_s))
longitude_list = list(map(float, longitude_list_s))

coordinates_list = list(zip(timestamp_list, latitude_list, longitude_list))
coordinates_list = list(set(coordinates_list))
coordinates_list.sort()
coordinates_list = list(zip(*coordinates_list))

timestamp_list = coordinates_list[0]
latitude_list = coordinates_list[1]
longitude_list = coordinates_list[2]

plt.scatter(longitude_list, latitude_list, color='blue', label='given')
tck, u = interpolate.splprep([longitude_list, latitude_list], s=0.0)
x_i, y_i = interpolate.splev(np.linspace(0, 1, 100), tck)

plt.plot(x_i, y_i, color='green', label='calculated')
plt.legend()
plt.show()
