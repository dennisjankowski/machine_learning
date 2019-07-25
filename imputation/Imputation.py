import numpy as np

n = 5
arr = np.random.uniform(high=6, size=(n, n))
for _ in range(3):
    arr[np.random.randint(n), np.random.randint(n)] = np.nan
print(arr)
print(20*'_')

#np.array([[0.25288643, 1.8149261 , 4.79943748, 0.54464834, np.nan],
#          [4.44798362, 0.93518716, 3.24430922, 2.50915032, 5.75956805],
#          [0.79802036, np.nan, 0.51729349, 5.06533123, 3.70669172],
#          [1.30848217, 2.08386584, 2.29894541, np.nan, 3.38661392],
#          [2.70989501, 3.13116687, 0.25851597, 4.24064355, 1.99607231]])

import impyute as impy
print(impy.fast_knn(arr))
print(20*'_')
print(impy.mean(arr))
print(20*'_')
