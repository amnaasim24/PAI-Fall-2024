
import numpy as np

array = np.random.randint(1, 200, 25)
percentile_75 = np.percentile(array, 75)

print("Array:\n", array)
print("75th Percentile:", percentile_75)
