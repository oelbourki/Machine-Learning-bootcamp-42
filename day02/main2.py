import numpy as np
from sigmoid import sigmoid_
from vec_log_loss import vec_log_loss_
# x = 4
# y_true = 1
# theta = 0.5
# y_pred = sigmoid_(x * theta)
# m = 1 
# # length of y_true is 1
# print(vec_log_loss_(y_true, y_pred, m))    # 0.1269280110429715


x = np.array([1, 2, 3, 4])
y_true = 0
theta = np.array([-1.5, 2.3, 1.4, 0.7])
y_pred = sigmoid_(np.dot(x, theta))
m = 1
print(vec_log_loss_(y_true, y_pred, m))

# x_new = np.arange(1, 13).reshape((3, 4))
# y_true = np.array([1, 0, 1])
# theta = np.array([-1.5, 2.3, 1.4, 0.7])
# y_pred = sigmoid_(np.dot(x_new, theta))
# m = len(y_true)
# print(vec_log_loss_(y_true, y_pred, m)) 