import numpy as np 
from sigmoid import sigmoid_
from vec_log_gradient import vec_log_gradient_
from vec_log_loss import vec_log_loss_
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
class LogisticRegressionBatchGd(object):
	def __init__(self, alpha=0.001, max_iter=1000, verbose=False, learning_rate='constant'):
		self.alpha = alpha
		self.max_iter = max_iter
		self.verbose = verbose
		self.learning_rate = learning_rate  # can be 'constant' or 'invscaling'
		self.thetas = []
		self.model = LogisticRegression(max_iter=max_iter,verbose=1,n_jobs=-1)
		# Your code here (e.g. a list of loss for each epochs...)
	def fit(self, x_train, y_train):
		"""
		Fit the model according to the given training data.
		Args:
			x_train: a 1d or 2d numpy ndarray for the samples
			y_train: a scalar or a numpy ndarray for the correct labels
		Returns: 
			self : object
			None on any error.
		Raises:
			This method should not raise any Exception.
		"""
		# self.model.fit(x_train, y_train)
		# self.thetas = np.hstack((self.model.intercept_[:,None], self.model.coef_)).reshape(-1,1)
		# return self.thetas
		# Your code here
		thetas = np.zeros((x_train.shape[1] + 1,1))
		np.insert(x_train, 0, 1,axis=1)
		
		m = x_train.shape[0]
		for i in range(5):
		    # print("cost: {}".format(vec_log_loss_(y_train,y_pred,m)),end='\r')
			y_pred = sigmoid_(x_train.dot(thetas))
			grad = vec_log_gradient_(x_train,y_train,y_pred)
			thetas = thetas - self.alpha * 0.5 * (1./m) * grad
			if i % 150 == 0 and self.verbose == True:
			    print("epoch {}     : loss {}".format(i,vec_log_loss_(y_train,y_pred,m)))4
			self.thetas = thetas
		return self.thetas
	def predict(self, x_train):
		"""
		Predict class labels for samples in x_train.
		Arg:
			x_train: a 1d or 2d numpy ndarray for the samples
		Returns: 
			y_pred, the predicted class label per sample.
			None on any error.
		Raises:
			This method should not raise any Exception.
		"""
		# Your code here
		#without intercept column
		x_train = np.insert(x_train, 0, 1,axis=1)
		return sigmoid_((x_train.dot(self.thetas)))
	def score(self, x_train, y_train):
		"""
		Returns the mean accuracy on the given test data and labels.
		Arg:
			x_train: a 1d or 2d numpy ndarray for the samples
			y_train: a scalar or a numpy ndarray for the correct labels
		Returns:
			Mean accuracy of self.predict(x_train) with respect to y_true
			None on any error.
		Raises:
			This method should not raise any Exception.
		"""
		# Your code here
		y_pred = self.predict(x_train)
		m = y_train.shape[0]
		return sum([1 if round(float(a)) == b else 0 for a,b in zip(y_pred,y_train)]) / m
	# def accuracy_score_(self,y_true, y_pred):
	# 	return sum([1 if round(float(a)) == b else 0 for a,b in zip(y_pred,y_true)])
	def accuracy_score_(self, y_true,y_pred):
		m = y_pred.shape[0]
		return (sum([1 if a == b else 0 for a,b in zip(y_pred,y_true)]) / m)
	def precision_score_(self, y_true,y_pred):
		tp = sum([1 if a == 1 and b == 1 else 0 for a,b in zip(y_pred,y_true)])
		fp = sum([1 if a == 1 and b == 0 else 0 for a,b in zip(y_pred,y_true)])
		return tp /(tp + fp)
	def recall_score_(self, y_true,y_pred):
		tp = sum([1 if a == 1 and b == 1 else 0 for a,b in zip(y_pred,y_true)])
		fn = sum([1 if a == 0 and b == 1 else 0 for a,b in zip(y_pred,y_true)])
		return tp /(tp + fn)
	def f1_score_(self, y_true,y_pred):
		"""
		F1 = 2 * (precision * recall) / (precision +    recall)
		"""
		precision = self.precision_score_(y_true,y_pred)
		recall = self.recall_score_(y_true,y_pred)
		return (2 * (precision * recall) / (precision + recall))