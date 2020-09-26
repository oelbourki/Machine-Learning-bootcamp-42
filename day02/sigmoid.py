import numpy as np 

def dot_list(lst, scalar):
    return [scalar * x for x in lst]

def sigmoid_(x):
    if isinstance(x,list):
        return 1. / (1 + np.exp(dot_list(x,-1)))
    return 1. / (1. + np.exp(x * -1))

if __name__ == "__main__":
    x = -4
    # print(dot_list(5,-1))
    # print(np.exp(-1 * [-4,2,0]))
    print(sigmoid__(5))