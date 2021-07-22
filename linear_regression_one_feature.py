import matplotlib.pyplot as plt
import numpy as np
from mlxtend.data import boston_housing_data

X, Y = boston_housing_data()
X = X[:, 7]              # DIS distance to 5 Boston employment centres
vec = np.ones(X.shape[0])
print(vec.shape)

temp = X

X = np.array([ vec, temp ]).T

print(X.shape)
print(Y.shape)

W = np.array([
    1,
    2,
    ])

print(W.shape)

plt.plot(X[:, 1], Y, 'bx')
plt.plot(X[:, 1], np.dot(X, W))
plt.show()

alpha = 0.00001
error = []
epochs = 50

for epoch in range(0, epochs):
    
    h = np.dot(X, W)
    J = 0.5 * np.sum( (h - Y)**2 )
    print(J)
    error.append(J)
    
    W = W - alpha * np.dot( X.T, (h-Y) )

    if epoch % 10 == 0:
        plt.plot(X[:, 1], Y, 'bx')
        plt.plot(X[:, 1], np.dot(X, W))
        plt.show()

plt.plot(range(0, epochs), error)
plt.show()


