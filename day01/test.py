import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
from mylinearregression import MyLinearRegression as MyLR 
from sklearn.metrics import mean_squared_error

df = pd.read_csv('./resources/are_blue_pills_magics.csv')
Xpill = np.array(df['Micrograms']).reshape((-1,1))
Yscore = np.array(df['Score']).reshape(-1,1)
linear_model1 = MyLR(np.array([[89.0], [-8]]))
linear_model2 = MyLR(np.array([[89.0], [-6]]))

def mse(y, y_hat):
	if y.shape != y_hat.shape:
		return None
	#return sum__(y_hat - y,lambda x: x**2)
	acc = 0.
	for elem1,elem2 in zip(y,y_hat):
		acc += (elem1 - elem2)**2
	return acc / y.shape[0]
Y_model1 = linear_model1.predict_(Xpill)
Y_model2 = linear_model2.predict_(Xpill)
print(linear_model1.mse_(Xpill,Yscore))
# print(mse(Xpill,Yscore))
print(mean_squared_error(Yscore, Y_model1))