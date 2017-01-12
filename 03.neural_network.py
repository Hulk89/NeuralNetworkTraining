import matplotlib.pyplot as plt
import numpy as np
from data import getData, showData

X, y, K = getData()
N, D = X.shape

# hyperparameters
step_size = 1e-0
reg = 1e-3

# initialize parameters randomly
h = 100
W = 0.01 * np.random.randn(D, h)
b = np.zeros((1, h))
W2 = 0.01 * np.random.randn(h, K)
b2 = np.zeros((1, K))

for i in range(10000):
    O = np.dot(X, W) + b
    hidden = np.maximum(0, O)
    scores = np.dot(hidden, W2) + b2
    # forward pass
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    correct_logprobs = -np.log(probs[range(N),y])
    data_loss = np.sum(correct_logprobs)/(N)
    reg_loss = 0.5*reg*(np.sum(W*W) + np.sum(W2*W2))
    loss = data_loss + reg_loss

    if i % 1000 == 0:
        print("iteration {}: loss {}".format(i, loss))
    # backward
    dscores = probs
    dscores[range(N),y] -= 1
    dscores /= N

    dW2 = np.dot(hidden.T, dscores)
    db2 = np.sum(dscores, axis=0, keepdims=True)
    dW2 += reg*W2
    
    dhidden = np.dot(dscores, W2.T)
    dO = dhidden
    dO[hidden <= 0] = 0
    dW = np.dot(X.T, dO)
    db = np.sum(dO, axis=0, keepdims=True)
    dW += reg*W

    W2 += -step_size * dW2
    b2 += -step_size * db2
    W += -step_size * dW
    b += -step_size * db

# evaluate training set accuracy
hidden_layer = np.maximum(0, np.dot(X, W) + b)
scores = np.dot(hidden_layer, W2) + b2
predicted_class = np.argmax(scores, axis=1)


print('training accuracy: %.2f' % (np.mean(predicted_class == y)))