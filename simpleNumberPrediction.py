from tensorflow import keras
import numpy as np


'''
This NN learns simple relationships between two series of numbers. 
In this case, the second series of numbers is always three times the first. 
This relationship is recognized so that new inputs (to_predict) can be predicted based on the model.
'''

model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')

xs=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
ys=[3, 6, 9, 12, 15, 18, 21, 24, 27, 30]

model.fit(xs, ys, epochs=4000)

to_predict = 10
result = model.predict([to_predict])
rounded_result = np.round(result[0][0], 2)

print(50 * '-')

print(rounded_result)
