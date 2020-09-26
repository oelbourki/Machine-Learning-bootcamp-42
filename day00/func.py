import numpy as np
import math

def sum__(x, f=lambda x: x):
	acc = 0.
	for item in x:
		acc += f(item) 
	return acc
def mean(x):
	n = x.shape[0]
	acc = 0.
	for item in x:
		acc += item
	return acc/n

def	variance(x):
	u = mean(x)
	return sum__(x,lambda x: (x - u)**2) / x.shape[0]

def std(x):
	return math.sqrt(variance(x))

def dot(x, y):
	if (x is None or y is None
	or x.shape[0] != y.shape[0]):
		return None
	return sum__(x * y, lambda x:x)

def mat_vec_prod(x, y):
	if (x.shape[1], 1) != y.shape:
		return None
	res = np.zeros((x.shape[0],1))
	f = 0
	for row in x:
		row = row.reshape((row.shape[0],1))
		res[f] = dot(row, y)
		f += 1
	return res
	
def mat_mat_prod(x, y):
	if (x.shape[0] != y.shape[1]):
		return None
	res = np.zeros((x.shape[0],y.shape[1]))
	print(x.shape," ",y.shape)
	for i in range(x.shape[0]):
		for j in range(y.shape[1]):
			res[i][j] = dot(x[i],y[:,j])
	return res

def mse(y, y_hat):
	if y.shape != y_hat.shape:
		return None
	#return sum__(y_hat - y,lambda x: x**2)
	acc = 0.
	for elem1,elem2 in zip(y,y_hat):
		acc += (elem1 - elem2)**2
	return acc / y.shape[0]

def vec_mse(y, y_hat):
	y = y.reshape((y.shape[0],1))
	y_hat = y_hat.reshape((y_hat.shape[0],1))
	# print(y.shape)
	return float(dot(y_hat - y,y_hat - y) / y_hat.shape[0])

def reshape(x):
	x = x.reshape(x.shape[0], 1)
	return x
def linear_mse(x, y, theta):
	theta = reshape(theta)
	y = reshape(y)
	hypothes = mat_vec_prod(x,theta)
	return float(mse(hypothes, y))


def	vec_linear_mse(x, y, theta):
	y = reshape(y)
	theta = reshape(theta)
	hypothes = mat_vec_prod(x, theta)
	return float(dot(hypothes - y,hypothes - y) / x.shape[0])
# print(vec_linear_mse(X, Y, W))


# def	gradient(x, y, theta):
# 	y = reshape(y)
# 	theta = reshape(theta)
# 	hypothes = mat_vec_prod(x, theta)
# 	# scalar = float(sum__(hypothes - y) / x.shape[0])
# 	grad =  np.zeros(theta.shape)
# 	for j in range(theta.shape[0]):
# 		for i in range(x.shape[0]):
# 			grad[j] = sum__((hypothes[i] - y[i]) * x[i])
# 	return grad



def vec_gradient(x, y, theta):
	y = reshape(y)
	theta = reshape(theta)
	return dot(x, mat_vec_prod(x, theta) - y) / x.shape[0]


# print(gradient(X, Y, Z))
# print(vec_gradient(X, Y, Z))