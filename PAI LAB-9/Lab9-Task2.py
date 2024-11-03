# Programming for AI LAB - FALL 2024
# Lab - 9

import matplotlib.pyplot as plt

cities = ['City A', 'City B', 'City C', 'City D', 'City E']
populations = [1200000, 850000, 500000, 3000000, 1500000]

plt.figure(figsize=(10, 6))
plt.barh(cities, populations, color='skyblue')
plt.title('Population of Cities')
plt.xlabel('Population')
plt.ylabel('Cities')
plt.grid(axis='x')

plt.tight_layout()
plt.show()
