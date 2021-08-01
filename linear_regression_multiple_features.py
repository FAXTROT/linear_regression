import matplotlib.pyplot as plt
import numpy as np

X = np.array(     # one feature feet
    [
    [1,10, 10**2],   # second feature is square of first
    [1,16, 16**2],
    [1,21, 21**2],
    [1,30, 30**2],
    [1,35, 35**2],
    [1,40, 40**2],
    [1,48, 48**2],
    [1,50, 50**2],
    [1,55, 55**2],
    [1,63, 63**2],
    [1,80, 80**2],
        ]
        )

Y = np.array([        # in thousands of US dollars
    20,
    14,
    60,
    72,
    79,
    87,
    90,
    112,
    142,
    186,
    225,
    ])

W = np.array([
    2,
    -1,
    0,
    ])

print(X.shape)
print(W.shape)
print(Y.shape)
print()

plt.plot(X[:, 1], Y, 'bx')
plt.plot(X[:, 1], np.dot(X, W))
plt.xlabel("feet")
plt.ylabel("price in 000 of US dollars")
plt.title("Initial dataset")
plt.show()

m, n = X.shape

alpha = 0.000000005
lmbda = 300000000
error = []
epochs = 10

for epoch in range(0, epochs):
    
    h = np.dot(X, W)
    J = 0.5 * np.sum( (h - Y)**2 )
    print(J)
    error.append(J)
    
    #W = W - alpha * np.dot( X.T, (h-Y) )
    W = W * ( 1 - alpha*lmbda/m ) - alpha * np.dot( X.T, (h-Y) )

    if epoch % 1 == 0:
        plt.plot(X[:, 1], Y, 'bx')
        plt.plot(X[:, 1], np.dot(X, W))
        plt.xlabel("feet")
        plt.ylabel("price in 000 of US dollars")
        plt.title(f"epoch number : {epoch}")
        plt.show()

plt.plot(range(0, epochs), error)
plt.xlabel("epochs")
plt.ylabel("error")
plt.title("J number")
plt.show()

