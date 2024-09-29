# Programming for AI LAB - FALL 2024
# Lab - 6

import pandas as pd

movies = {
  'Title': ['movie 1', 'movie 2', 'movie 3', 'movie 4', 'movie 5', 'movie 6'],
  'Revenue': [1.8, 2.7, 4.9, 0.8, 3.2, 2.2],
  'Budget': [1, 1.3, 0.7, 0.5, 0.9, 0.8]
}

df = pd.DataFrame(movies)
filtered_movies = df[(df['Revenue'] > 2) & (df['Budget'] < 1)]
print(filtered_movies)