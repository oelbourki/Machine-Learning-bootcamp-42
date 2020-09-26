import numpy as np 


def sum__(x, f=lambda x: x):
	acc = 0.
	for item in x:
		acc += f(item) 
	return acc
def mean(x):
	n = x.shape[0]
	acc = 0.
	for item in x:
		acc += item
	return acc/n
def dot(x, y):
	# if (x is None or y is None
	# or x.shape[0] != y.shape[0]):
	# 	return None
	return sum([a * b for a,b in zip(x,y)])

def vec_log_gradient_(x, y_true, y_pred):
    if (isinstance(y_pred, (int, float)) and
    isinstance(y_true, (int, float))):
        error = y_pred - y_true
        return x.dot(error)
    error = y_pred - y_true
    return (x.transpose().dot(error))
    # return -1. * (1. / m) * (x.transpose().dot(error))