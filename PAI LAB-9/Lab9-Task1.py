# Programming for AI LAB - FALL 2024
# Lab - 9

import matplotlib.pyplot as plt
import numpy as np

students = ['Student ' + str(i) for i in range(1, 21)]
heights = [150, 160, 165, 170, 155, 180, 175, 160, 168, 158,
           162, 170, 172, 158, 165, 174, 160, 159, 167, 173]

plt.figure(figsize=(10, 5))

colors = np.random.rand(len(students), 3)

plt.bar(students, heights, color=colors)
plt.title('Student Heights Bar Chart')
plt.xlabel('Students')
plt.ylabel('Height (cm)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 8))
plt.pie(heights, labels=students, autopct='%1.1f%%')
plt.title('Student Heights Pie Chart')
plt.axis('equal')
plt.show()
