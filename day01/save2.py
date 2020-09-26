import func
import numpy as np
from time import sleep
# X1 = np.array([[0.], [1.], [2.], [3.], [4.]])
# theta1 = np.array([[2.], [4.]])
# Y1 = np.array([[2.], [7.], [12.], [17.], [22.]])
# X3 = np.array([[0.2, 2., 20.], [0.4, 4., 40.], [0.6, 6., 60.], [0.8, 8.,80.]])
# X2 = np.array([[1], [2], [3], [5], [8]])
# theta2 = np.array([[2.]])
# theta3 = np.array([[0.05], [1.], [1.], [1.]])
X1 = np.array([[0.], [1.], [2.], [3.], [4.]])
theta1 = np.array([[1.], [1.]])
Y1 = np.array([[2.], [6.], [10.], [14.], [18.]])

def predict__(theta, X):
	print(theta.shape)
	print(X.shape)
	if (X.shape[1], 1) != theta.shape:
		print("incompatible dimension match between X and theta")
		return None
	return func.mat_vec_prod(X,theta)

# print(predict__(theta3,X3))



def cost_elem__(theta, X, Y):
	y_pred = predict__(theta, X)
	print(y_pred.shape)
	print(Y.shape)
	M = X.shape[0]
	return 0.5 * M * (y_pred - Y) ** 2

def cost__(theta1, X1, Y1):
	y_pred = predict__(theta1, X1)
	return func.vec_mse(y_pred, Y1) * 0.5

# print(cost_elem__(theta1, X1, Y1))
# print(cost__(theta1, X1, Y1))


# def ft_vec_gradient(x, y, theta):
# 	M = x.shape[0]
# 	N = x.shape[1]
# 	add = np.ones((x.shape[0],1))
# 	fix = np.concatenate((add, x),axis=1)
# 	# print(theta.shape)
# 	# print(x.shape)
# 	# print(y.shape)
# 	hp = predict__(theta, fix) - y
# 	if (M,1) != y.shape or (N,1) != theta.shape:
# 		print("incompatible")
# 		return None
# 	y = func.reshape(y)
# 	theta = func.reshape(theta)
# 	return func.dot(x, hp) / x.shape[0]


def	fit__(theta, X, Y,alpha=0.01,n_cycle=2000):
	M = X.shape[0]
	add = np.ones((X.shape[0],1))
	fix = np.concatenate((add, X),axis=1)
	for _ in range(n_cycle):
		hp = predict__(theta, fix) - Y
		grad = func.dot(fix, hp) * 0.5 * M
		theta = theta - alpha * 0.5 * M * func.vec_gradient(fix, Y, theta)
		# theta[0] = tmp0
	return theta
# add = np.ones((X1.shape[0],1))
# print(add.shape)
# print(type(add))

# print(X1.shape)
# print(np.concatenate((add, X1),axis=1))
print(fit__(theta1, X1,Y1))
# print(predict__(theta1, X1))