import numpy as np 

def vec_reg_linear_grad(x, y,theta, lambda_):
    m = x.shape[0]
    x_t = x.transpose()
    error = x.dot(theta) - y
    nabela = x_t.dot(error) / m 
    # print(nabela)
    nabela[1:] = nabela[1:] + theta[1:] * (lambda_ / m)
    return nabela


if __name__ == "__main__":
    X = np.array([
        [ -6, -7, -9],
        [ 13, -2, 14],
        [ -7, 14, -1],
        [ -8, -4, 6],
        [ -5, -9, 6],
        [ 1, -5, 11],
        [ 9, -11, 8]])
    Y = np.array([2, 14, -13, 5, 12, 4, -19])
    Z = np.array([3,10.5,-6])

    print(vec_reg_linear_grad(X,Y, Z, 1))