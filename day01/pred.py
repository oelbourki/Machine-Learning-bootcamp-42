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
X2 = np.array([[0.2, 2., 20.], [0.4, 4., 40.], [0.6, 6., 60.], [0.8, 8.,80.]])
Y2 = np.array([[19.6], [-2.8], [-25.2], [-47.6]])
theta2 = np.array([[42.], [1.], [1.], [1.]])


def predict__(theta, X):
	if (X.shape[1] + 1, 1) != theta.shape:
		print("incompatible dimension match between X and theta")
		return None
	return func.mat_vec_prod(X,theta[1:]) + theta[0]

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

def mat_vec_prod(x, y):
	# if (x.shape[0], 1) != y.shape:
	# 	print("p")
	# 	return None
	return (y.transpose()).dot(x)

def	fit__(theta, X, Y,alpha=0.00005,n_cycle=42000):
	M = X.shape[0]
	add = np.ones((X.shape[0],1),dtype=float)
	fix = np.concatenate((add, X),axis=1)
	for _ in range(n_cycle):
		hp = fix.dot(theta)
		error = hp - Y
		grad = (fix.transpose()).dot(error)
		tmp1 = theta - alpha * (0.5 / M) * grad
		theta = tmp1
	return theta
print()



print(fit__(theta2, X2,Y2))
# add = np.ones((X1.shape[0],1))
# fix = np.concatenate((add, X1),axis=1)
# print(fix)
# print(theta1)
# hp = fix.dot(theta1)
# error = hp - Y1
# grad = (fix.transpose()).dot(error)
# print(grad)
# print(hp.shape)
# print(fix.shape)
# tmp = mat_vec_prod(fix,hp)
# print(tmp)
# print(fit__())
# tmp1 = theta - alpha * M *0.5* func.mat_mat_prod(fix,hp)
# print(predict__(theta1, X1))