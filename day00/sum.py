import numpy as np
import math

W = np.array([[ -8, 8, -6, 14, 14, -9, -4],
	[ 2, -11, -2, -11, 14, -2, 14],
	[-13, -2, -5, 3, -8, -4, 13],
	[ 2, 13, -14, -15, -14, -15, 13],
	[ 2, -1, 12, 3, -7, -3, -6]])
Z = np.array([
[ -6, -1, -8, 7, -8],
 [ 7, 4, 0, -10, -10],
 [ 7, -13, 2, 2, -11],
 [ 3, 14, 7, 7, -4],
 [ -1, -3, -8, -4, -14],
 [ 9, -14, 9, 12, -7],
 [ -9, -4, -10, -3, 6]])
X = np.array([0, 15, -9, 7, 12, 3, -21])
Y = np.array([2, 14, -13, 5, 12, 4, -19])
# X = np.array([0, 15, -9, 7, 12, 3, -21]).reshape((7,1))
# Y = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((7,1))
def sum__(x, f):
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

# print(mat_vec_prod(W, X))
# print(W.dot(X))
# print(mean(X**2))
		
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
	acc = 0.
	for elem1,elem2 in zip(y,y_hat):
		acc += (elem1 - elem2)**2
	return acc / y.shape[0]

def vec_mse(y, y_hat):
	y = y.reshape((y.shape[0],1))
	y_hat = y_hat.reshape((y_hat.shape[0],1))
	# print(y.shape)
	return float(dot(y_hat - y,y_hat - y) / y_hat.shape[0])

def linear_mse(x, y, theta):
	return mse(dot(theta, x), y)
# print(X.shape)
# print(X)
# print(X.reshape((1,X.shape[0])))
# print(X)

# print(linear_mse(X, Y))