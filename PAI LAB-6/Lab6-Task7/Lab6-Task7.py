# Programming for AI LAB - FALL 2024
# Lab - 6

import pandas as pd

df = pd.read_excel('employee.xlsx', engine='openpyxl')

specified_year = 2020
employees_in_year = df[df['Year'] == specified_year]
print(employees_in_year)
