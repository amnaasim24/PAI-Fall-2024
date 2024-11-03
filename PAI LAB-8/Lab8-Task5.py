# Programming for AI LAB - FALL 2024
# Lab - 8

import numpy as np

choices = [2, 5, 9, 10]
matrix_a = np.random.choice(choices, size=(4, 4))

identity_matrix = np.eye(4)
result_matrix = np.dot(matrix_a, identity_matrix)

odd_matrix = np.array([[1, 3, 5, 7],
                       [9, 11, 13, 15],
                       [17, 19, 21, 23],
                       [25, 27, 29, 31]])

final_matrix = result_matrix + odd_matrix

print("Matrix A:\n", matrix_a)
print("\nResult after multiplication with identity matrix:\n", result_matrix)
print("\nFinal matrix after addition:\n", final_matrix)
