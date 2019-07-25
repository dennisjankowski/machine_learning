import csv

import dateutil

print(__doc__)

# Author: Vincent Dubourg <vincent.dubourg@gmail.com>
#         Jake Vanderplas <vanderplas@astro.washington.edu>
#         Jan Hendrik Metzen <jhm@informatik.uni-bremen.de>s
# License: BSD 3 clause

import numpy as np
from matplotlib import pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


# ----------------------------------------------------------------------
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

print(len(coordinates_list))


# copy the coordinates list and remove x percent of entries
coordinates_list_less = coordinates_list[:]

# sort the lists by timestamps and unzip the triples into three seperate lists
coordinates_list.sort()
coordinates_list = list(zip(*coordinates_list))

timestamp_list = coordinates_list[0]
latitude_list = coordinates_list[1]
longitude_list = coordinates_list[2]

latitude_list = np.atleast_2d(latitude_list).T


# Instantiate a Gaussian Process model
kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)

print("x2:", latitude_list)
print("y2:", longitude_list)
# Fit to data using Maximum Likelihood Estimation of the parameters
gp.fit(latitude_list, longitude_list)

# Make the prediction on the meshed x-axis (ask for MSE as well)
y_pred, sigma = gp.predict(latitude_list, return_std=True)

# Plot the function, the prediction and the 95% confidence interval based on
# the MSE
plt.figure()
plt.plot(latitude_list, longitude_list, 'r:', label=r'$f(x) = x\,\sin(x)$')
plt.plot(latitude_list, longitude_list, 'r.', markersize=10, label='Observations')
plt.plot(latitude_list, y_pred, 'b-', label='Prediction')

plt.xlabel('$x$')
plt.ylabel('$f(x)$')
# plt.ylim(-10, 20)
plt.legend(loc='upper left')
plt.show()
