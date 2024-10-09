# Programming for AI LAB - FALL 2024
# Lab - 8

import numpy as np

even_matrix = np.array([[2, 4, 6], [8, 10, 12], [14, 16, 18]])
result_matrix = even_matrix * 4

print("Resultant Matrix after multiplying with 4:")
print(result_matrix)

identity_matrix = np.eye(3)

print("\n3x3 Identity Matrix:")
print(identity_matrix)

final_matrix = np.dot(result_matrix, identity_matrix)

print("\nFinal Matrix after multiplying with Identity Matrix:")
print(final_matrix)
