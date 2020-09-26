import numpy as np 

def reg_mse(y, x, theta, lambda_):
	error = x.dot(theta) - y
	m = x.shape[0]
	return float((1. / m) * (error.dot(error)
	+ lambda_ * theta[1:].dot(theta[1:])))


if __name__ == "__main__":
	X = np.array([
		[ -6, -7, -9],
	[ 13, -2, 14],
	[ -7, 14, -1],
	[ -8, -4, 6],
	[ -5, -9, 6],
	[ 1, -5, 11],
	[ 9, -11, 8]])
	Y = np.array([2, 14, -13, 5, 12, 4, -19])
	Z = np.array([3,0.5,-6])
	print(reg_mse(Y,X,Z,0))