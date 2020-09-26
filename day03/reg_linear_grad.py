import numpy as np 



def reg_linear_grad(y, x,theta, lambda_, alpha=1):
	print(x.shape)
	m = x.shape[0]
	n = x.shape[1]
	nabela = np.zeros((n,1))
	acc = 0
	for i in range(m):
		nabela[0] += (((x[i].dot(theta) - y[i]) * x[i][0])) / m
	for j in range(1,n):
		acc = 0
		for i in range(m):
			acc += ((x[i].dot(theta) - y[i]) * x[i][j])
		nabela[j] = (acc + lambda_ * theta[j]) / m
	return nabela
if __name__ == "__main__":
	X = np.array([
	[ -6, -7, -9],
	[ 13, -2, 14],
	[ -7, 14, -1],
	[ -8, -4, 6],
	[ -5, -9, 6],
	[ 1, -5, 11],
	[ 9, -11, 8]])
	print(X.shape)
	Y = np.array([2, 14, -13, 5, 12, 4, -19])
	Z = np.array([3,10.5,-6])
	print(reg_linear_grad(Y, X, Z, 1))