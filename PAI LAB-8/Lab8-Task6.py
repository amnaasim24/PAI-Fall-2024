# Programming for AI LAB - FALL 2024
# Lab - 8

import numpy as np

matrix_5x3 = np.random.randint(1, 10, size=(5, 3))

matrix_3x2 = np.random.randint(1, 10, size=(3, 2))

result_matrix = np.dot(matrix_5x3, matrix_3x2)

print("5x3 Matrix:\n", matrix_5x3)
print("\n3x2 Matrix:\n", matrix_3x2)
print("\nResult of multiplication (5x2 Matrix):\n", result_matrix)
