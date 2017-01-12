import matplotlib.pyplot as plt
import numpy as np
from data import getData, showData

X, y, K = getData()
N, D = X.shape

# hyperparameters
step_size = 1e-0
reg = 1e-3
delta = 1
# initialize parameters randomly
W = 0.01 * np.random.randn(D, K)
b = np.zeros((1, K))

for i in range(200):
    scores = np.dot(X, W) + b
    # forward pass
    correctScore = scores[range(N), y] # (N, )
    correctScoreMat = correctScore.T * np.ones( scores.T.shape )
    correctScoreMat = correctScoreMat.T

    marginMat = np.ones(scores.shape) * delta
    L = np.maximum(scores - correctScoreMat + marginMat, 0)
    L[range(N), y] = 0

    data_loss = np.sum(L)/N

    reg_loss = 0.5*reg*np.sum(W*W)
    loss = data_loss + reg_loss

    if i % 10 == 0:
        print("iteration {}: loss {}".format(i, loss))
    # backward
    dscores = np.ones(L.shape)
    dscores[L == 0] = 0
    dscores /= N

    dW = np.dot(X.T, dscores)
    db = np.sum(dscores, axis=0, keepdims=True)
    dW += reg*W

    W += -step_size * dW
    b += -step_size * db

# evaluate training set accuracy
scores = np.dot(X, W) + b
predicted_class = np.argmax(scores, axis=1)  # (N, )

print('training accuracy: %.2f' % (np.mean(predicted_class == y)))
