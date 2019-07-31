import csv
import random

import dateutil
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
import numpy as np

'''
This NN learns simple relationships between two series of numbers. 
In this case, the second series of numbers is always three times the first. 
This relationship is recognized so that new inputs (to_predict) can be predicted based on the model.
'''

def remove_random_entries(l, n):
    for _ in range(int(len(l) * n)):
        l.pop(random.randrange(0, len(l)))
    return l


def normalize_data(series):
    values = series.reshape((len(series), 1))
    # train the normalization
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(values)
    print('Min: %f, Max: %f' % (scaler.data_min_, scaler.data_max_))
    # normalize the dataset and print the first 5 rows
    normalized = scaler.transform(values)
    return normalized


def denormalize_data(normalized_data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    inversed = scaler.inverse_transform(normalized_data)
    return inversed


model = keras.Sequential([keras.layers.Dense(units=500, input_shape=[2])])
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

latitude_list = np.array(latitude_list)
longitude_list = np.array(longitude_list)

coordinates_list = list(zip(timestamp_list, latitude_list, longitude_list))
coordinates_list = list(set(coordinates_list))


# copy the coordinates list and remove x percent of entries
coordinates_list_less = coordinates_list[:]

# sort the lists by timestamps and unzip the triples into three seperate lists
coordinates_list.sort()
coordinates_list = remove_random_entries(coordinates_list, 0.6)
coordinates_list = coordinates_list[-100:]
coordinates_list = list(zip(*coordinates_list))

timestamp_list = coordinates_list[0]
latitude_list = coordinates_list[1]
longitude_list = coordinates_list[2]

time_list = []
counter = 0
for x in range(100):
    time_list.append(counter)
    counter = counter + 1

timestamp_list = np.array(time_list)
latitude_list = np.array(latitude_list)
longitude_list = np.array(longitude_list)

print(latitude_list)

latitude_list = normalize_data(latitude_list)
longitude_list = normalize_data(longitude_list)
timestamp_list = normalize_data(timestamp_list)

input_list = np.array(list(zip(latitude_list, time_list)))

print(input_list)

xs1 = latitude_list
xs2 = timestamp_list
ys = longitude_list

model.fit(input_list, ys, epochs=10)


to_predict_0 = [53.94, 74]
test = [to_predict_0]
test = np.array(test)
result_0 = model.predict(test)

#print('result:', result_0[0][0])

import matplotlib.pyplot as plt

plt.plot(xs1, ys, 'ro', color='blue')
#plt.plot(to_predict_0[0], [result_0[0][0]], 'ro')
plt.show()

print(50 * '-')

#print(rounded_result_0)
#print(rounded_result_1)
#print(rounded_result_2)
