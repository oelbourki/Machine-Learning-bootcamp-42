import numpy as np

def log_gradient_(x, y_true, y_pred):
	print(type(y_pred))
	print(type(y_true))

	if (isinstance(y_true, (list,np.ndarray)) and isinstance(y_pred, (list,np.ndarray))):
		error = y_pred - y_true
		nabela = [0] * len(x[0])
		print(len(x[0][:]))
		print(x[0])
		for i in range(len(x[0])):
			# print(x[:])
			s = x[i][:]
			nabela[i] = sum([a * b for a,b in zip(s,error)])
		return nabela
	error = y_pred - y_true
	# print(error)
	nabela = [0] * len(x)
	# print(nabela)
	for i in range(len(x)):
		x[i] *= error
	return x

	# 	nabela
	
	# return sum([(a - b) * c for a, b, c in zip()])



if __name__ == "__main__":
	X = np.array([[ -6, -7, -9],
		[ 13, -2, 14],
		[ -7, 14, -1],
		[ -8, -4, 6],
		[ -5, -9, 6],
		[ 1, -5, 11],
		[ 9, -11, 8]])
	Y = np.array([2, 14, -13, 5, 12, 4, -19])
	Z = np.array([3,0.5,-6])