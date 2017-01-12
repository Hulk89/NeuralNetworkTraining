import matplotlib.pyplot as plt
import numpy as np

def getData():
    N = 100 # class마다 점의 갯수
    D = 2 # point의 dimension
    K = 3 # class 갯수

    X = np.zeros((N*K, D)) # data matrix (각 row가 data point)
    y = np.zeros(N*K, dtype='uint8') # class label들

    for j in range(K):
        ix = range(N*j, N*(j+1))
        r = np.linspace(0.0, 1, N) # radius
        t = np.linspace(j*4, (j+1)*4, N) + np.random.randn(N)*0.2 # theta

        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)] # concatenation along the second axis.
        y[ix] = j
    return X, y, K

def showData(X, y):
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    plt.show()
