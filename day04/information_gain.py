import numpy as np 
from gini import gini
from entropy import entropy
# from scipy.stats import entropy



def information_gain(array_source, array_children_list, criterion='gini'):
	# try:
	# 	if isinstance(array_,np.ndarray) and array_.size == 0:
	# 		return None
	# except:
	# 	return None
	# if (isinstance(array_, list)):
	# 	array_ = np.array(array_)
	# N = float(array_.size)
	# unique , counts= np.unique(array_,return_counts=True)
	# val = dict(zip(unique,counts))
	# dct = {}
	# acc = 0
	# for key in val.items():
	# 	pi = key[1] / N
	# 	acc += np.power(pi,2)
	# return 1 - acc
	G0 = entropy(array_source)
	N = array_source.size
	# print(N)
	# print(G0)
	
	q = len(array_children_list)
	iG = np.zeros((q,1))
	acc = 0
	iG[0] = entropy(array_children_list[0])
	# print(array_children_list[1].size / N)
	iG[1] = entropy(array_children_list[1]) 
	S = (array_children_list[0].size / N) * iG[0] + (array_children_list[1].size / N) * iG[1]
	# for i in range(q):
	# 	n = array_children_list[i].size
	# 	acc += (n/N) * entropy(array_children_list[i])
	# print()
	return float(G0 - S)

if __name__ == "__main__":
	# X = []
	X1 = np.arange(20).reshape(1,20)
	f = np.array([0., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
	f1 = np.array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]) 
	s = np.hstack((f ,f1))
	print(s)
	lst = [f,f1]
	print(information_gain(s,lst,))