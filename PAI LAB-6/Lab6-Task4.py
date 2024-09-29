# Programming for AI LAB - FALL 2024
# Lab - 6

import pandas as pd

data = {
    'name': ['Amna', 'Momina', 'Anum', 'Barirah', 'Warisha', 'Riya',
             'Rija', 'Emaan', 'Areeba'],
    'score': [12.5, 9, 16.5, None, 9, 20, 14.5, None, 8],
    'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2],
    'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no',]
}

df = pd.DataFrame(data)
# Set a custom index starting from 100
df.index = range(100, 100 + len(df))
print(df)
