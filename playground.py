import numpy as np

a = np.arange(0, 1000000000)
b = np.arange(0, 500)
a = a.reshape(50, 20, 10, 1000, 10, 10)
b = b.reshape(50, 1, 10, 1, 1, 1)

import time

start = time.perf_counter()
c = a + b
end = time.perf_counter()
elapsed = end - start 

print(elapsed)