
import numpy as np

matrix = np.array([[1, 2], [3, 4]])
determinant = round(np.linalg.det(matrix), 2)
if determinant != 0:
    inverse = np.linalg.inv(matrix)
else:
    inverse = "Undefined"

print("Matrix:\n", matrix)
print("Determinant:", determinant)
print("Inverse:\n", inverse)
