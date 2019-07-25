from tensorflow import keras
import numpy as np


'''
This NN learns simple relationships between two series of numbers. 
In this case, the second series of numbers is always three times the first. 
This relationship is recognized so that new inputs (to_predict) can be predicted based on the model.
'''

model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')

# data series without gap from 5 to 7
xs=[1, 2, 3, 4, 8, 9, 10]
ys=[4, 8, 12, 16, 32, 36, 40]

model.fit(xs, ys, epochs=4000)

to_predict_0 = 5
result_0 = model.predict([to_predict_0])
rounded_result_0 = np.round(result_0[0][0], 2)

to_predict_1 = 6
result_1 = model.predict([to_predict_1])
rounded_result_1 = np.round(result_1[0][0], 2)

to_predict_2 = 7
result_2 = model.predict([to_predict_2])
rounded_result_2 = np.round(result_2[0][0], 2)

print(50 * '-')

print(rounded_result_0)
print(rounded_result_1)
print(rounded_result_2)

