import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import dateutil.parser
import csv
import random


def remove_random_entries(l, n):
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

print(len(coordinates_list))


# copy the coordinates list and remove x percent of entries
coordinates_list_less = coordinates_list[:]

# sort the lists by timestamps and unzip the triples into three seperate lists
coordinates_list.sort()
coordinates_list = list(zip(*coordinates_list))

timestamp_list = coordinates_list[0]
latitude_list = coordinates_list[1]
longitude_list = coordinates_list[2]

# plt.scatter(longitude_list, latitude_list, color='blue', label='given')
tck, u = interpolate.splprep([longitude_list, latitude_list], s=0.0)
x_i, y_i = interpolate.splev(np.linspace(0, 1, 100), tck)
plt.plot(x_i, y_i, color='red', label='all points')

# -----------------------------------------------------------------------------

# copy the coordinates list and remove x percent of entries
remove_random_entries(coordinates_list_less, 0.80)
coordinates_list_less.sort()

print(len(coordinates_list_less))

coordinates_list_gap = []
for j in range(len(coordinates_list_less)):
    if not 40 < j < 140:
        coordinates_list_gap.append(coordinates_list_less[j])

print(len(coordinates_list_gap))


# coordinates_list_less = list(zip(*coordinates_list_less))
coordinates_list_gap = list(zip(*coordinates_list_gap))

coordinates_list_less = coordinates_list_gap

timestamp_list_less = coordinates_list_less[0]
latitude_list_less = coordinates_list_less[1]
longitude_list_less = coordinates_list_less[2]

plt.scatter(longitude_list_less, latitude_list_less, color='green', label='given')
tck, u = interpolate.splprep([longitude_list_less, latitude_list_less], s=0.0)
x_i_2, y_i_2 = interpolate.splev(np.linspace(0, 1, 100), tck)

plt.plot(x_i_2, y_i_2, color='blue', label='less points')

# 80% less points than the original data, and a 31% data gap (60 and 120)
# 80% less points than the original data, and a 52% data gap (40 and 140)

plt.legend()
plt.show()
