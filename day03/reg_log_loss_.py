import numpy as np 



def dot_list(lst, scalar):
	return [scalar * x for x in lst]

def sigmoid_(x):
	if isinstance(x,list):
		return 1. / (1 + np.exp(dot_list(x,-1)))
	return 1. / (1 + np.exp(x * -1))

def reg_log_loss_(y_true, y_pred, m,  theta, lambda_,eps= 1e-15):
	
	# lf = np.log(y_pred)
	# lt = np.log(1. - y_pred)
	# y_ft = y_true.transpose()
	# y_tt = (1. - y_true).transpose()
	# p = y_ft.dot(lf) + y_tt.dot(lt) 
	# return (-1. * (1./m)) * p
	
	
	
	# y_pred = np.clip(y_pred, eps,1 - eps)
	y_tt = 1. * y_true.transpose()
	y_ttt = (1. - y_true).transpose()
	lf = np.log(y_pred)
	lt = np.log(1.-y_pred)
	p1 = y_tt.dot(lf) 
	p2 =  y_ttt.dot(lt)
	theta = np.atleast_1d(theta)
	return (p1 - p2 + lambda_ * theta.dot(theta)) / m


if __name__ == "__main__":
	x_new = np.arange(1, 13).reshape((3, 4))
	y_true = np.array([1, 0, 1])
	theta = np.array([-1.5, 2.3, 1.4, 0.7])
	y_pred = sigmoid_(np.dot(x_new, theta))
	m = len(y_true)
	print(reg_log_loss_(y_true,y_pred, m, theta, 0.0))
	# 7.233346147374828`