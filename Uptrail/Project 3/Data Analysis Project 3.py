import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

from scipy.stats import pearsonr, zscore, f_oneway
from factor_analyzer.factor_analyzer import calculate_kmo, FactorAnalyzer
from matplotlib.colors import LinearSegmentedColormap
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

'''
///
Note: I don't plan to reuse this code so I won't organise it.
///
'''


path = 'Insert data path here.'

'''///Load Data///'''

data = pd.read_excel(path)
#print(data.describe())
data = data[(zscore(data[['Total_Spend', 'Purchase_Frequency', 'Marketing_Spend', 'Seasonality_Index', 'Average_Purchase_Value', 'Customer_Lifetime_Value']]) < 3)]
#print(data.describe())



'''///Correlation Matrix///

quantitative_values = ['Total_Spend', 'Purchase_Frequency', 'Marketing_Spend', 'Seasonality_Index', 'Average_Purchase_Value', 'Customer_Lifetime_Value']
corr_data = data[quantitative_values].corr()
gradient = LinearSegmentedColormap.from_list('rg',["r", "w", "g"], N=256) 
p_values = round(corr_data.corr(method=lambda x, y: pearsonr(x, y)[1]), 4)
mask = np.invert(np.tril(p_values < 0.05))
ax, fig = plt.subplots(figsize=(6, 6))


#print(p_values)
#print(corr_data)


sns.heatmap(corr_data, mask=mask, cmap=gradient, annot=True, annot_kws={'fontweight' : 'bold'}, linewidths=2, linecolor='black', square=True, vmax=1, vmin=-1)

plt.xticks(rotation=25)
plt.title('Correlation Matrix')
plt.show()
'''



'''///(Kaiser-Meyer-Olkin) KMO Test///

kmo_model = calculate_kmo(data[quantitative_values])

for x in range(len(kmo_model[0])):
    print(f'{quantitative_values[x]} KMO Score: {kmo_model[0][x]:.4f}')
'''



'''///Factor Analysis///

features = ['Customer_Lifetime_Value', 'Total_Spend', 'Purchase_Frequency', 'Seasonality_Index', 'Marketing_Spend']

fa_model = FactorAnalyzer(n_factors=2, rotation='varimax')
fa_model.fit(data[features])

eigen_values, v = fa_model.get_eigenvalues()

print('Eigen Values: ')
for x in range(len(eigen_values)):
    print(f'Factor {x}:  {eigen_values[x]:.4f}')

print('Factor Loadings: ')
loadings = pd.DataFrame(fa_model.loadings_, index=features)
print(loadings.round(2))
'''




'''///ANOVA One-way Test///

north = data[data['Region'] == 'North']['Customer_Lifetime_Value']
east = data[data['Region'] == 'East']['Customer_Lifetime_Value']
south = data[data['Region'] == 'South']['Customer_Lifetime_Value']
west = data[data['Region'] == 'West']['Customer_Lifetime_Value']

anova = f_oneway(north, east, south, west)
print('Anova P-Value for Regions & CLV: ', round(anova.pvalue, 6))
'''


''' ///Post-hoc Boxplots of ANOVA Test///

sns.boxplot(x=data['Customer_Lifetime_Value'], y=data['Region'], palette='Dark2', hue=data['Region'])

plt.title('Distribution of CLV by Regions')
plt.xlabel("")
plt.ylabel("")
plt.show()
'''



'''///Logistic Regression Graph///'''

data['Churned_Binary'] = data['Churned'].map({'No' : 0, 'Yes' : 1})
#print(data['Churned'])

#Seperate to training (75%) & testing data (25%).
random_no = data[data['Churned_Binary'] == 0].sample(2, random_state=1)
random_yes = data[data['Churned_Binary'] == 1].sample(2, random_state=1)

test_data = pd.concat([random_no, random_yes])
training_data = data.drop(test_data.index)

#print(training_data)
#print(test_data)

#Double brackets for 2D array for fit(). X and y and training data for the model.
X_clv = training_data[['Customer_Lifetime_Value']].values
X_total = training_data[['Total_Spend']].values
X_market = training_data[['Marketing_Spend']].values


scaler = StandardScaler()
''' For normalized graph.
X_clv = scaler.fit_transform(data[['Customer_Lifetime_Value']].values)
X_total = scaler.fit_transform(data[['Total_Spend']].values)
X_market = scaler.fit_transform(data[['Marketing_Spend']].values)
'''


y = training_data['Churned_Binary']

#print(X)
#print(y)

model_clv = LogisticRegression()
model_clv.fit(X_clv, y)

model_total = LogisticRegression()
model_total.fit(X_total, y)

model_market = LogisticRegression()
model_market.fit(X_market, y)


#Generate a series  of (500) evenly spaced points from min and max x values as test data. Then convert it into a multiple row (-1) by single column (1) array. Greater intervals improves line smoothness.
x_range_clv = np.linspace(X_clv.min(), X_clv.max(), 500).reshape(-1, 1)
#Use the model to predict y-axis values for each x-axis value (500).
y_probs_clv = model_clv.predict_proba(x_range_clv)[:, 1]


x_range_total = np.linspace(X_total.min(), X_total.max(), 500).reshape(-1, 1)
y_probs_total = model_total.predict_proba(x_range_total)[:, 1]


x_range_market = np.linspace(X_market.min(), X_market.max(), 500).reshape(-1, 1)
y_probs_market = model_market.predict_proba(x_range_market)[:, 1]


#print(x_range)
#print(y_probs)


''''''
plt.plot(x_range_clv, y_probs_clv, color='blue', linewidth=1, label='CLV')
plt.scatter(X_clv, y, color="blue", marker='x', alpha=0.3)

plt.plot(x_range_total, y_probs_total, color='green', linewidth=1, label='Total Spend')
plt.scatter(X_total, y, color="green", marker='x', alpha=0.3)

plt.plot(x_range_market, y_probs_market, color='red', linewidth=1, label='Marketing Spend')
plt.scatter(X_market, y, color="red", marker='x', alpha=0.3)

plt.title('Probability of Churn by Value')
plt.xlabel('Value (£)')
plt.ylabel('Probability of Churn')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()




'''///Steepness of normalised regression lines///'''

scaler = StandardScaler()
X_clv_scaled = scaler.fit_transform(X_clv)
X_total_scaled = scaler.fit_transform(X_total)
X_market_scaled = scaler.fit_transform(X_market)

model_clv_std = LogisticRegression().fit(X_clv_scaled, y)
model_total_std = LogisticRegression().fit(X_total_scaled, y)
model_market_std = LogisticRegression().fit(X_market_scaled, y)

X_purchase = training_data[['Purchase_Frequency']].values
X_season = training_data[['Seasonality_Index']].values

X_purchase_scaled = scaler.fit_transform(X_purchase)
X_season_scaled = scaler.fit_transform(X_season)

model_purchase_std = LogisticRegression().fit(X_purchase_scaled, y)
model_season_std = LogisticRegression().fit(X_season_scaled, y)

print('Normalized β₁:')
print(f"CLV β₁: {abs(model_clv_std.coef_[0][0]):.4f}")
print(f"Total Spend β₁: {abs(model_total_std.coef_[0][0]):.4f}")
print(f"Marketing Spend β₁: {abs(model_market_std.coef_[0][0]):.4f}")
print(f"Purchase Frquency β₁: {abs(model_purchase_std.coef_[0][0]):.4f}")
print(f"Seasonality Index β₁: {abs(model_season_std.coef_[0][0]):.4f}")




'''///Confusion Matrix///'''

y_test = test_data['Churned_Binary']
y_predict = (model_clv.predict_proba(test_data[['Customer_Lifetime_Value']])[:, 1] >= 0.5).astype(int)
#y_predict = (model_total.predict_proba(test_data[['Total_Spend']])[:, 1] >= 0.5).astype(int)
#y_predict = (model_market.predict_proba(test_data[['Marketing_Spend']])[:, 1] >= 0.5).astype(int)
confusion = confusion_matrix(y_test, y_predict)
sns.heatmap(confusion, cmap='Greens', square=True)
plt.title('Confusion Matrix Evaluating CLV Logistic Regression using Test Data')
plt.show()




'''///Customer Segmentation w/KMeans, PCA, and Normalization///

features = ['Customer_Lifetime_Value', 'Total_Spend', 'Purchase_Frequency', 'Seasonality_Index', 'Marketing_Spend']
cluster_label = ['High Value Customers', 'Low Value Customers', 'Mid Value Customers']

k_data = data[features]
k_data_scaled = scaler.fit_transform(k_data)

kmeans = KMeans(n_clusters=3, random_state=1)
data['Customer_Segment'] = kmeans.fit_predict(k_data_scaled)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(k_data_scaled)


centroids = pca.fit_transform(kmeans.cluster_centers_)

print(pd.DataFrame(pca.components_, columns=features, index=['PCA Component 1', 'PCA Component 2']))

cluster_summary = data.groupby('Customer_Segment')[features].mean().round(2)
print(cluster_summary)


for clusters in range(3):
    plt.scatter(X_pca[data['Customer_Segment'] == clusters, 0], X_pca[data['Customer_Segment'] == clusters, 1], alpha=0.5, label=f'{cluster_label[clusters]}')

plt.scatter(centroids[:, 0], centroids[:, 1], cmap='red', marker='x', label='Centroid', alpha=0.8, linewidths=3, s=100)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('Customer Segmentation with KMeans and Normalised PCA Values')
plt.legend()
plt.show()
'''



'''///Decision Trees Ensemble Learning///

X = data[features]
y = data['Customer_Segment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

tree_model = DecisionTreeClassifier(max_depth=5, random_state=1)
tree_model.fit(X_train, y_train)

y_pred = tree_model.predict(X_test)


print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


plot_tree(tree_model, feature_names=features, class_names=cluster_label)
plt.title("Decision Tree for KMeans-Based Customer Segmentation")
plt.show()
'''




'''///Linear Regression Model for Marketing Spend & Seasonality Index///

linear_x_training = training_data[['Marketing_Spend']].values
linear_y_training = training_data['Seasonality_Index']

linear_x_test = test_data[['Marketing_Spend']].values
linear_y_test = test_data['Seasonality_Index']

linearModel = LinearRegression()
linearModel.fit(linear_x_training, linear_y_training)

linear_predict_y = linearModel.predict(linear_x_training)

#Couldn't figure out a solution to extend the line whilst retaining confidence index.
sns.regplot(x=training_data['Marketing_Spend'], y=linear_y_training, label='Linear Regression', color='blue', marker='x', line_kws={'color': 'green'}, scatter=False)
sns.scatterplot(x=training_data['Marketing_Spend'], y=linear_y_training, label='Test Data', color='red')
sns.scatterplot(x=test_data['Marketing_Spend'], y=linear_y_test, label='Training Data', color='blue')
plt.title('Linear Regression: Marketing Spend by Seasonality Index')
plt.xlabel('Marketing Spend (£)')
plt.ylabel('Seasonality Index')
plt.legend()
plt.show()

#Determine the accuracy of the model. R2 = Strength of linear relationship between variables (0 to 1) - [High = Strong]. MSE = Fit of predicted to actual values - [Low = Tight]
r2 = r2_score(linear_y_training, linear_predict_y)
mse = mean_squared_error(linear_y_training, linear_predict_y)

print(f"R² Score: {r2:.4f}")
print(f"Mean Squared Error: {mse:.4f}")
'''
