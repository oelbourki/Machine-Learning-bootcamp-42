import numpy as np 



def minmax(x):
	'''
	x m*1
	'''
	min_ = np.min(x)
	max_ = np.max(x)
	x_ = np.zeros_like(x,dtype=float)
	for i in range(x.shape[0]):
		x_[i] = (x[i] - min_) / (max_ - min_) 
	return x_ 


if __name__ == "__main__":
	X = np.array([0, 15, -9, 7, 12, 3, -21])
	print(minmax(X))