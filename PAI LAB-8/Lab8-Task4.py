# Programming for AI LAB - FALL 2024
# Lab - 8

import numpy as np

dtype = [('name', 'U10'), ('height', 'f4'), ('class', 'i4')]

students = np.array([('Amna', 5.3, 10),('Momina', 5.5, 9),('Anum', 5.1, 10),('Barirah', 5.0, 9)], dtype=dtype)

sorted_students = np.sort(students, order=['class', 'height'])
print(sorted_students)
