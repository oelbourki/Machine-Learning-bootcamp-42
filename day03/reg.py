import numpy as np 

def regularization(theta, lambda_):
    return lambda_ * sum([a**2 for a in theta[1:]])


if __name__ == "__main__":
    X = np.array([0, 15, -9, 7, 12, 3, -21])
    print(regularization(X,0.3))
    