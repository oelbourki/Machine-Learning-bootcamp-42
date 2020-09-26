import numpy as np 
from sklearn.preprocessing import PolynomialFeatures 


def polynomialFeatures(x, degree=2, interaction_only= False,include_bias=True):
    Poly = PolynomialFeatures(degree, interaction_only,include_bias)
    return Poly.fit_transform(x)

if __name__ == "__main__":
    X = np.arange(6).reshape(3, 2)
    print(X)
    print(polynomialFeatures(X,degree=3,interaction_only=False))