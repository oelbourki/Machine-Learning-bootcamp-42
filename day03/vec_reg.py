import numpy as np 

def vectorized_regularization(theta, lambda_):
    return lambda_ * theta[1:].dot(theta[1:])


if __name__ == "__main__":
    X = np.array([0, 15, -9, 7, 12, 3, -21])
    print(vectorized_regularization(X,0.3))
    