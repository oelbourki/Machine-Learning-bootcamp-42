import numpy as np
from mylinearregression import MyLinearRegression as MyLR
X = np.array([[1., 1., 2., 3.], [5., 8., 13., 21.], [34., 55., 89.,144.]])
Y = np.array([[23.], [48.], [218.]])
theta = np.array([[1.], [1.], [1.], [1.], [1]])
print("X",X.shape)
print("Y",Y.shape)
print("theta",theta.shape)

# MyLR().
mylr = MyLR(theta)
mylr.predict_(X)
mylr.cost_elem_(X,Y)
mylr.cost_(X,Y)
mylr.fit_(X, Y, alpha = 1.6e-4, n_cycle=200000)
mylr.theta
mylr.predict_(X)
mylr.cost_elem_(X,Y)
mylr.cost_(X,Y)
