# from func import * 
import math

def log_loss_(y_true, y_pred, m, eps=1e-15):
	if (isinstance(y_true,list) or isinstance(y_pred,list)):
		acc = 0.
		for i in range(m):
			p1 = y_true[i] * math.log(y_pred[i])
			p2 = (1 - y_true[i]) * math.log(1 - y_pred[i])
			acc += p1 + p2 
		return float(-1. * (1. / m) * acc) 
	# if (isinstance(y_true,int) or isinstance(y_pred,int)):
	acc = 0
	p1 = y_true * math.log(y_pred)
	p2 = (1 - y_true) * math.log(1 - y_pred)
	acc += p1 + p2 
	return float(-1. * (1. / m) * acc) 
	# if (y_pred.shape != y_true.shape):
		# return -1
	# print(y_true.shape)
	# print(y_pred.shape)


