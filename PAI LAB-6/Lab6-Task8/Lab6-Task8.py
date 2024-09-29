# Programming for AI LAB - FALL 2024
# Lab - 6

import pandas as pd
import random
from datetime import datetime, timedelta

product_data = {
    'ProductID': range(1, 51),
    'ProductName': [f'Product {i}' for i in range(1, 51)],
    'Category': ['Electronics', 'Clothing', 'Home', 'Books', 'Sports'] * 10,
    'Price': [round(20 + (i * 1.5), 2) for i in range(50)]
}
products_df = pd.DataFrame(product_data)
products_df.to_csv('products.csv', index=False)

order_data = {
    'Order ID': range(1, 101),
    'ProductID': [random.randint(1, 50) for _ in range(100)],
    'Quantity': [random.randint(1, 5) for _ in range(100)],
    'Order Date': [(datetime.now() - timedelta(days=random.randint(0, 30))).strftime('%m-%d-%Y') for _ in range(100)]
}
orders_df = pd.DataFrame(order_data)
orders_df.to_csv('orders.csv', index=False)

products_df = pd.read_csv('products.csv')
orders_df = pd.read_csv('orders.csv')

print("Products DataFrame (First 5 Rows):")
print(products_df.head())

print("\nOrders DataFrame (First 5 Rows):")
print(orders_df.head())

print("\nProducts DataFrame (Last 10 Rows):")
print(products_df.tail(10))

print("\nOrders DataFrame (Last 10 Rows):")
print(orders_df.tail(10))

orders_df['Total Revenue'] = orders_df['Quantity'] * products_df.loc[orders_df['ProductID'] - 1, 'Price'].values
total_revenue = orders_df['Total Revenue'].sum()
print(f"\nTotal Revenue Generated from All Orders: ${total_revenue:.2f}")

best_selling = orders_df.groupby('ProductID').agg({'Quantity': 'sum'}).nlargest(5, 'Quantity')
low_selling = orders_df.groupby('ProductID').agg({'Quantity': 'sum'}).nsmallest(5, 'Quantity')

print("\nTop 5 Best-Selling Products:")
print(best_selling)

print("\nTop 5 Low-Selling Products:")
print(low_selling)

top_product_ids = best_selling.index
top_categories = products_df[products_df['ProductID'].isin(top_product_ids)]['Category']
most_common_category = top_categories.mode()[0]

print(f"\nMost Common Category Among Top 5 Best-Selling Products: {most_common_category}")

average_price_by_category = products_df.groupby('Category')['Price'].mean()
print("\nAverage Price of Products in Each Category:")
print(average_price_by_category)

orders_df['Order Date'] = pd.to_datetime(orders_df['Order Date'])

daily_revenue = orders_df.groupby(orders_df['Order Date']).agg({'Total Revenue': 'sum'})
highest_revenue_date = daily_revenue['Total Revenue'].idxmax()
highest_revenue_value = daily_revenue['Total Revenue'].max()

highest_day = highest_revenue_date.day
highest_month = highest_revenue_date.month
highest_year = highest_revenue_date.year

print(f"\nHighest Revenue Day: {highest_day}, Month: {highest_month}, Year: {highest_year}")

monthly_revenue = orders_df.groupby(orders_df['Order Date'].dt.to_period('M')).agg({'Total Revenue': 'sum'}).reset_index()
monthly_revenue.columns = ['Month', 'Total Revenue']
print("\nTotal Revenue for Each Month:")
print(monthly_revenue)

null_values_products = products_df.isnull().sum()
null_values_orders = orders_df.isnull().sum()

print("\nNull Values in Products DataFrame:")
print(null_values_products)

print("\nNull Values in Orders DataFrame:")
print(null_values_orders)

if null_values_products.any() or null_values_orders.any():
    products_df.dropna(inplace=True)
    orders_df.dropna(inplace=True)
    print("\nCleaned DataFrames (Dropped Nulls if any).")