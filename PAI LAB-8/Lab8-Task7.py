# Programming for AI LAB - FALL 2024
# Lab - 8

import numpy as np

random_array = np.random.rand(1000)

average = np.mean(random_array)
variance = np.var(random_array)
std_dev = np.std(random_array)

with open("textfile.txt", "w") as file:
  file.write(f"Average: {average}\nVariance: {variance}\nStandard Deviation: {std_dev}")
