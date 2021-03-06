import matplotlib.pyplot as plt
import numpy as np

X = np.array([
    [1,10,],
    [1,16,],
    [1,21,],
    [1,30,],
    [1,35,],
    [1,40,],
    [1,48,],
    [1,50,],
        ])

Y = np.array([
    20,
    14,
    60,
    72,
    79,
    87,
    90,
    88,
    ])

W = np.array([   # init weights
    0,
    0,
    ])

print(X.shape)
print(W.shape)
print(Y.shape)
print()

plt.plot(X[:, 1], Y, 'bx')
plt.title(f"Init points")
plt.show()

alpha = 0.00001
error = []
epochs = 50

for epoch in range(0, epochs):
    
    h = np.dot(X, W)
    J = 0.5 * np.sum( (h - Y)**2 )
    print(f"MSE = {J}")
    error.append(J)
    
    W = W - alpha * np.dot( X.T, (h-Y) )

    if epoch % 10 == 0:
        plt.plot(X[:, 1], Y, 'bx')
        plt.plot(X[:, 1], np.dot(X, W))
        plt.title(f"Epoch № {epoch}")
        plt.show()

plt.plot(range(0, epochs), error)
plt.title(f"Error (MSE)")
plt.show()

