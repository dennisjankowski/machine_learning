from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt

def inputfunct(x):
    return 0.25*(np.sin(2*np.pi*x*x)+2.0)

np.random.seed(5)
X = np.random.sample([2048])
Y = inputfunct(X) + 0.2*np.random.normal(0,0.2,len(X))

Xreal = np.arange(0.0, 1.0, 0.01)
Yreal = inputfunct(Xreal)


### Model creation: adding layers and compilation
model = Sequential()
model.add(Dense(8, input_dim=1, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='linear'))

model.compile(optimizer='adam', loss='mse', metrics=['mse'])


nepoch = 1000
nbatch = 32
model.fit(X, Y, epochs=nepoch, batch_size=nbatch)


Ylearn = model.predict(Xreal)


print('test')

### Make a nice graphic!
plt.plot(X,Y,'.', label='Raw noisy input data')
plt.plot(Xreal,Yreal, label='Actual function, not noisy', linewidth=4.0, c='black')
plt.plot(Xreal, Ylearn, label='Output of the Neural Net', linewidth=4.0, c='red')
plt.legend()
plt.show()
plt.savefig('neural-network-keras-function-interpolation.png')

print('test2')

