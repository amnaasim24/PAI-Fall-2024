# Programming for AI LAB - FALL 2024
# Lab - 10

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

sns.set_theme()
data = pd.read_csv('heart.csv')
print('Shape of the data is ', data.head())


numeric_df = data.select_dtypes(include=['float64', 'int64'])
sns.set(font_scale=0.6)
plt.figure(figsize=(14, 12))
sns.heatmap(numeric_df.corr(),annot=True)


sns.kdeplot(x='chol' , data = numeric_df)


sns.histplot(x = 'sex', data = numeric_df)
plt.ylabel('frequency')


sns.histplot(x = 'chol',kde = True , data = numeric_df)
plt.ylabel('frequency')


sns.scatterplot(data=numeric_df , x = 'sex' , y = 'chol')


gender = ['Male', 'Female']
ratio = [68.3, 31.7]
plt.pie(ratio, labels=gender, colors=['skyblue', 'lightgreen'], autopct = "%1.1f%%")
plt.show()