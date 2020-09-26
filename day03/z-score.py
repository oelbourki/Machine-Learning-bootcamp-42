import numpy as np 



def zscore(x):
	'''
	x m*1
	'''
	mean = np.mean(x)
	std = np.std(x)
	x_ = np.zeros_like(x,dtype=float)
	for i in range(x.shape[0]):
		x_[i] = (x[i] - mean) / std 
	return x_ 


if __name__ == "__main__":
	X = np.array([0, 15, -9, 7, 12, 3, -21])
	print(zscore(X))