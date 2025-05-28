import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

path = 'Insert dataset path here'

data = pd.read_csv(path)
data['Order_Date'] = pd.to_datetime(data['Order_Date'])

gradient = LinearSegmentedColormap.from_list('rg',["r", "w", "g"], N=256) 



print(data[['Revenue', 'Discount (%)']].describe())
#print(data[['Revenue', 'Discount (%)', 'Order_Date']].corr())

#plt.matshow(data[['Revenue', 'Discount (%)', 'Order_Date']].corr())
#plt.show()



corr = data[['Revenue', 'Discount (%)', 'Order_Date']].corr()
p_values = round(corr.corr(method=lambda x, y: pearsonr(x, y)[1]), 4)
mask = np.triu(corr)

#With P-values
#mask = np.invert(np.tril(p_values<0.05))

print("P-Values: ")
print(p_values)

ax, fig = plt.subplots(figsize=(6, 6))

#Add mask=mask for masking.
#sns.heatmap(corr, cmap=gradient, annot=True, mask=mask, annot_kws={'fontweight':'bold'}, linecolor='black', linewidths=2, square=True, vmin=0, vmax=1)


#plt.title('Correlation Matrix')

plt.hist(data['Revenue'])

plt.show()