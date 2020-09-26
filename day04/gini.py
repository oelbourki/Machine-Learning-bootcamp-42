import numpy as np 



def gini(array_):
	try:
		if isinstance(array_,np.ndarray) and array_.size == 0:
			return None
	except:
		return None
	if (isinstance(array_, list)):
		array_ = np.array(array_)
	N = float(array_.size)
	unique , counts= np.unique(array_,return_counts=True)
	val = dict(zip(unique,counts))
	dct = {}
	acc = 0
	for key in val.items():
		pi = key[1] / N
		acc += np.power(pi,2)
	return 1 - acc



if __name__ == "__main__":
	# X = []
	X1 = np.arange(20).reshape(1,20)
	f = [0., 1., 1., 1., 1., 1., 1., 1., 1., 1.] 
	m = [0,0,1]
	print(gini(m))