
import numpy as np

A = np.array([[2, 3, 1], [4, 1, -2], [-1, 2, 3]])
b = np.array([1, -2, 3])

solution = np.round(np.linalg.solve(A, b), 3)
print("Solution (x, y, z):", solution)
