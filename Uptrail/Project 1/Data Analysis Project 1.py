import pandas as pd
import os

path = 'Insert dataset path here'

data = pd.read_excel(path)
data['Date'] = pd.to_datetime(data['Date'])

#print(data.info())
print(" ")
#print("The mean of [Price]: ", round(data['Price'].mean(), 2))
#print("The standard deviation of [Price]:", round(data['Price'].std(), 2))
print(round(data.describe(), 2))
print(" ")

print("Correlations:")
print(data.corr(numeric_only=True))

print(" ")



category = data.groupby('Category')

print('''The MODE of [Price] by [Category]:
      
''', category['Price'].count())

print(" ")

print('''The SUM of [Quantity] by [Category]:
      
''', category['Quantity'].sum())

print(" ")

print('''The MEAN of [Price] by [Category]:
      
''', round(category['Price'].mean(), 2))

print(" ")

print('''The STANDARD DEVIATION of [Price] by [Category]:
      
''', round(category['Price'].std(), 2))

print(" ")



categoryProduct = data.groupby(['Category', 'Product'])

print('''The MODE of [Price] by [Category] & [Product]:
      
''', categoryProduct['Price'].count())

print(" ")

print('''The SUM of [Quantity] by [Category] & [Product]:
      
''', categoryProduct['Quantity'].sum())

print(" ")

print('''The MEAN of [Price] by [Category] & [Product]:
      
''', round(categoryProduct['Price'].mean(), 2))

print(" ")

print('''The STANDARD DEVIATION of [Price] by [Category] & [Product]:
      
''', round(categoryProduct['Price'].std(), 2))

print(" ")



print('''Transactions with more than 1 product sold: 

''', data.loc[data['Quantity'] > 1, ['Category', 'Product', 'Quantity']])

print(" ")



month = data.groupby(data['Date'].dt.to_period('M'))
print('''Total revenue by MONTH:

''', month['Total_Amount'].sum())

