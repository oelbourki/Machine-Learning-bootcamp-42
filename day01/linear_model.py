import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler
from mylinearregression import MyLinearRegression as MyLR 

# df = pd.read_csv('./resources/are_blue_pills_magics.csv')
df = pd.read_csv('data.csv')

Y = df['price'].to_numpy().reshape((len(df['price']),1))
# print(Y)
X = df.drop(['price'],axis=1).values
scaler = MinMaxScaler()
scaler = scaler.fit(X)
X = scaler.transform(X)
print(X)
model = MyLR(np.zeros((X.shape[1] + 1,1)))
print(Y.shape)
print(X.shape)
model.fit_(X,Y,0.1)
Y_pred = model.predict_(X)
print(model.mse_(X,Y))
plt.plot( X,Y,'o')
plt.plot( X,Y_pred,'+')

plt.show()
# print(X.describe())
# print(Y)

