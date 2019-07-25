import os
import random

import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')

# Data taken from https://blog.dominodatalab.com/fitting-gaussian-process-models-python/

X, y = np.array([
     [-0.98,-0.25], [-0.87,-1.20], [-0.78,-0.49], [-0.68,-0.83], [-0.57,-0.15],
     [-0.50, 0.00], [-0.38,-1.10], [-0.29,-0.32], [-0.18,-0.60], [-0.09,-0.49],
     [0.03 ,-0.50], [0.09 ,-0.02], [0.20 ,-0.47], [0.31 ,-0.11], [0.41 ,-0.28],
     [0.53 , 0.40], [0.61 , 0.11], [0.70 , 0.32], [0.94 , 0.42], [1.02 , 0.57],
     [1.13 , 0.82], [1.24 , 1.18], [1.30 , 0.86], [1.43 , 1.11], [1.50 , 0.74],
     [1.63 , 0.75], [1.74 , 1.15], [1.80 , 0.76], [1.93 , 0.68], [2.03 , 0.03],
     [2.12 , 0.31], [2.23 ,-0.14], [2.31 ,-0.88], [2.40 ,-1.25], [2.50 ,-1.62]]).T


from pykrige import OrdinaryKriging

X_pred = np.linspace(-6, 6, 200)

# pykrige doesn't support 1D data for now, only 2D or 3D
# adapting the 1D input to 2D
uk = OrdinaryKriging(X, np.zeros(X.shape), y, variogram_model='gaussian',)

y_pred, y_std = uk.execute('grid', X_pred, np.array([0.]))

y_pred = np.squeeze(y_pred)
y_std = np.squeeze(y_std)

fig, ax = plt.subplots(1, 1, figsize=(10, 4))
ax.scatter(X, y, s=40, label='Input data')


ax.plot(X_pred, y_pred, label='Predicted values')
ax.fill_between(X_pred, y_pred - 3*y_std, y_pred + 3*y_std, alpha=0.3, label='Confidence interval')
ax.legend(loc=9)
ax.set_xlabel('x')
ax.set_ylabel('y')
#ax.set_xlim(-6, 6)
#ax.set_ylim(-2.8, 3.5)
if 'CI' not in os.environ:
    # skip in continous integration
    plt.show()