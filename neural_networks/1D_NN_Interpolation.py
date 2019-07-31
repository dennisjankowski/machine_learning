import csv
import random
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


def remove_random_entries(l, n):
    for _ in range(int(len(l) * n)):
        l.pop(random.randrange(0, len(l)))
    return l


def normalize_data(series):
    values = series.reshape((len(series), 1))
    # train the normalization
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(values)
    normalized = scaler.transform(values)
    return normalized


def denormalize_data(normalized_data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    inversed = scaler.inverse_transform(normalized_data)
    return inversed


def load_data(path):
    csv_file = csv.DictReader(open(file=path))
    latitude_list_s = []
    longitude_list_s = []

    for row in csv_file:
        longitude_list_s.append(float(row['longitude']))
        latitude_list_s.append(float(row['latitude']))

    latitude_list = list(map(float, latitude_list_s))
    longitude_list = list(map(float, longitude_list_s))

    latitude_list = np.array(latitude_list)
    longitude_list = np.array(longitude_list)

    coordinates_list = list(zip(latitude_list, longitude_list))
    coordinates_list = list(set(coordinates_list))

    # sort the lists by timestamps and unzip the triples into three seperate lists
    coordinates_list.sort()
    coordinates_list = remove_random_entries(coordinates_list, 0.6)
    coordinates_list = coordinates_list[-100:]
    coordinates_list = list(zip(*coordinates_list))

    latitude_list = coordinates_list[0]
    longitude_list = coordinates_list[1]

    latitude_list = np.array(latitude_list)
    longitude_list = np.array(longitude_list)

    latitude_list = normalize_data(latitude_list)
    longitude_list = normalize_data(longitude_list)

    return latitude_list, longitude_list


##################################################################################################

''' defines the model of the neural network '''
model = keras.Sequential()
model.add(keras.layers.Dense(units=50, input_shape=[1]))
model.add(keras.layers.Dense(units=1, input_shape=[1]))
model.compile(optimizer='adam', loss='mean_squared_error')

''' loads some trajectory data of a csv file and preprocess it'''
x, y = load_data('../resources/trajectory.csv')
model.fit(x, y, epochs=1000)

x_coord_list = []
y_coord_list = []
x_coordinate = 0.0
plot_step_size = 0.025

''' calculates the prediction of the model for some values in the distance of the plot_step_size'''
for i in range(40):
    result_0 = model.predict([x_coordinate])
    x_coordinate = x_coordinate + plot_step_size
    x_coord_list.append(x_coordinate)
    y_coord_list.append(result_0[0])

''' plots the graph'''
plt.plot(x, y, 'ro', color='blue')
plt.plot(x_coord_list, y_coord_list, 'ro')
plt.show()
