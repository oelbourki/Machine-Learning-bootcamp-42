import func
import numpy as np
from time import sleep
# X1 = np.array([[0.], [1.], [2.], [3.], [4.]])
# theta1 = np.array([[2.], [4.]])
# Y1 = np.array([[2.], [7.], [12.], [17.], [22.]])



X1 = np.array([[0.], [1.], [2.], [3.], [4.]])
theta1 = np.array([[1.], [1.]])
Y1 = np.array([[2.], [6.], [10.], [14.], [18.]])


def predict__(theta, X):
	print(X.shape)
	print(theta.shape)
	if (X.shape[1] + 1, 1) != theta.shape:
		print("incompatible dimension match between X and theta")
		return None
	return func.mat_vec_prod(X,theta[1:]) + theta[0]


def cost_elem__(theta, X, Y):
	y_pred = predict__(theta, X)
	# print(y_pred.shape)
	# print(Y.shape)
	M = X.shape[0]
	return 0.5 * M * (y_pred - Y) ** 2

def cost__(theta1, X1, Y1):
	y_pred = predict__(theta1, X1)
	return func.vec_mse(y_pred, Y1) * 0.5

# print(cost_elem__(theta1, X1, Y1))
# print(cost__(theta1, X1, Y1))



def	fit__(theta, X, Y,alpha=0.01,n_cycle=2000):
	M = X.shape[0]
	for _ in range(n_cycle):
		# print("cost {}".format(cost__(theta,X,Y)),end='\r')
		# hp = predict__(theta, X) - Y
		# tmp0 =alpha * M * func.dot(hp,hp)
		tmp = alpha * M * func.vec_gradient(X, Y, theta[1:])
		# theta[0] = tmp0
		theta = tmp
	return theta
print()
print(fit__(theta1, X1,Y1))
# print(predict__(theta1, X1))