import pandas as pd
import numpy as np
from node import Node
from gini import gini
from information_gain import information_gain

class DecisionTreeClassifier:
	def __init__(self, criterion='gini', max_depth=10):
		"""
		Args:
		criterion: str which should be in ['gini', 'entropy'].
			max_depth: max_depth of the tree (Decision tree creation stops splitting a node if node.depth >= max_depth).
		"""
		self.root = Node()  # Root node of the tree
		# Your code here. You can add more things if needed

	def fit(self, X, y):
		""" Build the decision tree from the training set (X, y).
	The training set has m data_points (examples).
		Each of them has n features.
		Args:
			X: a pandas.Dataframe representing the training input of dimension m x n.
			y: a pandas.Dataframe representing the labels (m x 1).
		Returns:
		 object self: Trained tree.
	Raises:
		This method should not raise any Exception.
		"""
		# Your code here. You can add more things if needed
		# self.root
		r = gini(X)
		# print(f'r>> {r}')
		# print(X.iloc[:,:])
		for feat in range(4):
			tmp = X.iloc[:,feat]
			mean = np.mean(tmp)
			right,left = self.split_(X,feat,mean)
			inf = information_gain(X,[right,left])
			print(inf)
		# print(inf)
		return self.root
	def split_(self,X,id,mean):
		t = X[id] >= mean
		return X[X[id] >= mean],X[X[id] < mean]
if __name__ == '__main__':
	from sklearn.model_selection import train_test_split
	from sklearn.datasets import load_iris 
	# sklearn is not allowed in the classes.

	# Test on iris dataset
	iris = load_iris()
	X = pd.DataFrame(iris.data)
	y = pd.DataFrame(iris.target)
	X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=1)
	# print(X_train)
	# print(y_train)

	dec_tree = DecisionTreeClassifier()
	dec_tree.fit(X_train, y_train)
	root = dec_tree.root
	# print(root)
	# print(X)
	# print("TEST ON IRIS DATASET")
	# print("Root split info = 'Feature_{}{}{}'\n".format(root.split_feature, root.split_kind, root.split_criteria))
	# print("5 first lines of the labels of the left child of root =\n{}\n".format(root.left_child.y.head()))
	# print("5 first lines of the labels of the right child of root =\n{}".format(root.right_child.y.head()))