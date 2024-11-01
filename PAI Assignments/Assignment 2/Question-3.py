
import numpy as np

arr = np.random.randint(1, 1000, 20)
print("A random 1-D Array: ", arr)
matrix = arr.reshape(4, 5)
print(f"Reshaped Array:\n{matrix}")
