import matplotlib.pyplot as plt
import numpy as np
from mlxtend.data import boston_housing_data

X, Y = boston_housing_data()
vec = np.ones( [X.shape[0], 1])
print(vec.shape)

temp = X

X = np.column_stack((temp, vec))

print(X.shape)
print(Y.shape)

W = np.random.rand(X.shape[1], 1)

print(W.shape)

alpha = 0.00000001
error = []
epochs = 50

for epoch in range(0, epochs):
    
    h = np.dot(X, W)
    J = 0.5 * np.sum( (h - Y)**2 )
    print(J)
    error.append(J)
    
    W = W - alpha * np.dot( X.T, (h-Y) )

    """
    if epoch % 10 == 0:
        plt.plot(X[:, 1], Y, 'bx')
        plt.plot(X[:, 2], np.dot(X, W))      # can not plot multidimensional thing
        plt.show()
    """


plt.plot(range(0, epochs), error)
plt.show()


