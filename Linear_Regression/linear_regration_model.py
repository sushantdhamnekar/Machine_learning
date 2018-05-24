import pandas as pd
import seaborn as sns


data = pd.read_excel('Regression_Beispiel.xlsx')


data.dropna()

#data.head()
data.info()
#data.columns

sns.pairplot(data)



X = data[['Monate', 'KM-Stand']]
y = data['Preis']

print('x and y', X,y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.0, random_state=101)

from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)



'''
predictions = lm.predict(X_test)
print('Predictions:',predictions)

plt.scatter(y_test,predictions)
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
'''

data1 = pd.read_excel('Regression_prediction.xlsx')
predictions1 = lm.predict(data1)
print('Predictions:',predictions1)

