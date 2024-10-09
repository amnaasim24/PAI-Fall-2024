# Programming for AI LAB - FALL 2024
# Lab - 8

import numpy as np

odd_num = np.arange(1, 18, 2)
odd_arr = odd_num.reshape(3, 3)
for row in odd_arr:
    for element in row:
        print(element, end=" ")
    print()
