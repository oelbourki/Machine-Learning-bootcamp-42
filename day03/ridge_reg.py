from mylinearregression import MyLinearRegression
import numpy as np 

class MyRidge(MyLinearRegression):
	# def __init__(self):
	# 	super.__init__()
	def get_params_(self):
		return self.theta

	def set_params_(self, theta):
		self.setTheta(theta)
	# def predict_(self):
	# 	self.pr
	def cost_elem_(self, X, Y):
		M = Y.shape[0]
		pred = X.dot(self.theta)
		return (np.power((pred - Y),2) + self.theta.dot(self.theta)) * (0.5 / M) 
	def cost_(self, X, Y):
		return float(sum(self.cost_elem_(X, Y)))
	def fit_(self, X, Y, alpha=0.01, lambda_=1.0, max_iter=1000, tol=0.001):
		print("traning........")
		M = Y.shape[0]
		X = self.concat(X)
		for _ in range(int(max_iter)):
			error = X.dot(self.theta) - Y
			grad = (X.transpose()).dot(error)
			self.theta = self.theta - alpha * (1. / M ) * 0.5 * grad
			print("cost: {}".format(self.cost_(X, Y)),end='\r')
		return self.theta