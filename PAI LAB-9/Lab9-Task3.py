# Programming for AI LAB - FALL 2024
# Lab - 9

import matplotlib.pyplot as plt

flavors = ['Vanilla', 'Chocolate', 'Strawberry', 'Mint', 'Cookie Dough']
scoops_sold = [150, 100, 80, 60, 90]

plt.figure(figsize=(8, 6))
plt.pie(scoops_sold, labels=flavors, autopct='%1.1f%%', startangle=140)
plt.title('Distribution of Ice Cream Sales')
plt.axis('equal')
plt.show()
