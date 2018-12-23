import numpy as np


def relu(x, deriv=False):
    if (deriv == False):
        return np.maximum(x, 0, x)
    x[x <= 0] = 0
    x[x > 0] = 1
    return x


X = np.random.randn(1, 3)
y = np.random.randn(1, 1)
np.random.seed(1)
w0 = np.random.randn(3, 6)
w1 = np.random.randn(6, 1)
b0 = np.random.randn(1, 6)
b1 = np.random.randn(1, 1)
for j in range(10000):
    l0 = X
    l1 = relu(np.dot(l0, w0)) + b0
    l2 = np.dot(l1, w1) + b1
    l2_error = ((y - l2) ** 2).mean()
    print("Error:" + str(l2_error))

    l2_delta = 2 * l2_error

    l1_error = l2_delta * (w1.T)
    l1_delta = l1_error * relu(l1, deriv=True)
    w1 += l1.T.dot(l2_delta) * 0.001
    w0 += l0.T.dot(l1_delta) * 0.001
