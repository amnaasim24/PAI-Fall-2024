
import numpy as np

array = np.random.randint(1, 600, 50)
max_index = np.argmax(array)
min_index = np.argmin(array)

print("Array:", array)
print("Index of Maximum Value:", max_index)
print("Index of Minimum Value:", min_index)
