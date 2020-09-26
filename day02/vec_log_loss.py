import numpy as np 

def vec_log_loss_(y_true, y_pred, m, eps=1e-15):
	if m != 1:
		lf = np.log(y_pred)
		lt = np.log(1. - y_pred)
		y_ft = y_true.transpose()
		y_tt = (1. - y_true).transpose()
		p = y_ft.dot(lf) + y_tt.dot(lt) 
		return (-1. * (1./m)) * p
	lf = np.log(y_pred)
	lt = np.log(1. - y_pred)
	# y_ft = np.atleast_1d(y_true)
	# y_tt = np.atleast_1d(1. - y_true)
	# p = y_ft.dot(lf) + y_tt.dot(lt) 
	p = y_true * lf + (1 - y_true) * lt 
	return (-1. * (1./m)) * p
if __name__ == "__main__":
	vec_log_loss_(0,0,0)