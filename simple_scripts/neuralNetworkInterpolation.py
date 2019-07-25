import csv
import random

import dateutil
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


def remove_random_entries(l, n):
    for _ in range(int(len(l) * n)):
        l.pop(random.randrange(0, len(l)))
    return l

'''
This NN learns simple relationships between two series of numbers. 
'''

model = keras.Sequential()
model.add(keras.layers.Dense(units=5000, input_shape=[1]))
model.add(keras.layers.Dense(units=10000, input_shape=[5000]))
model.add(keras.layers.Dense(units=15000, input_shape=[10000]))
#model.add(keras.layers.Dense(units=2000, input_shape=[5000], activation=keras.activations.softmax))

model.compile(optimizer='sgd', loss='mean_squared_error')

csv_file = csv.DictReader(open(file='../resources/trajectory.csv'))
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


# sort the lists by timestamps and unzip the triples into three seperate lists
coordinates_list.sort()
coordinates_list = remove_random_entries(coordinates_list, 0.9)
coordinates_list = coordinates_list[-50:]

coordinates_list = list(zip(*coordinates_list))

timestamp_list = coordinates_list[0]
latitude_list = coordinates_list[1]
longitude_list = coordinates_list[2]

xs = latitude_list
ys = longitude_list

model.fit(xs, ys, epochs=20)

to_predict_0 = 53.88
result_0 = model.predict([to_predict_0])
print(result_0)
rounded_result_0 = np.round(result_0[0][0], 2)

print(50 * '-')
print(rounded_result_0)

plt.plot(xs, ys, 'ro', color='blue')
plt.plot([to_predict_0], [rounded_result_0], 'ro')
plt.show()
