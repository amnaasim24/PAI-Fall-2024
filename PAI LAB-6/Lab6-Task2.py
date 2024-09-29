# Programming for AI LAB - FALL 2024
# Lab - 6

import pandas as pd

movies = {
    'Title': ['movie 1', 'movie 2', 'movie 3', 'movie 4', 'movie 5', 'movie 6'],
    'Runtime': [120, 150, 95, 110, 135, 105]
}

df = pd.DataFrame(movies)
sorted_movies = df.sort_values(by='Runtime', ascending=False)
print(sorted_movies)
