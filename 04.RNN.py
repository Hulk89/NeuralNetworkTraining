import numpy as np
from classesForNN import Gates, ActivationFunctions

mulGate = Gates.MultiplyGate()
addGate = Gates.AddGate()
activation = ActivationFunctions.Tanh()

class RNNCell:
    def forward(self, x, prev_s, Wxh, Whh, Why):
        self.mulu = mulGate.forward(Wxh, x)
        self.mulw = mulGate.forward(Whh, prev_s)
        self.add = addGate.forward(self.mulw, self.mulu)
        self.s = activation.forward(self.add)
        self.pred = mulGate.forward(Why, self.s)

    def backward(self, x, prev_s, U, W, V, diff_s, dPred):
        self.forward(x, prev_s, Wxh, Whh, Why)
        dWhy, dstmp = mulGate.backward(Why, self.s, dPred)
        ds = dstmp + diff_s
        dadd = activation.backward(self.add, ds)
        dmulw, dmulu = addGate.backward(self.mulw, self.mulu, dadd)
        dWhh, dprev_s = mulGate.backward(Whh, prev_s, dmulw)
        dWxh, dx = mulGate.backward(Wxh, x, dmulu)
        return (dprev_s, dU, dW, dV)

class Model:
    def __init__(self, word_dim, hidden_dim=100, bptt_truncate=4):
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        self.Wxh = np.random.uniform(-np.sqrt(1. / word_dim), np.sqrt(1. / word_dim), (hidden_dim, word_dim))
        self.Whh = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (hidden_dim, hidden_dim))
        self.Why = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (word_dim, hidden_dim))

    def forward_propagation(self, x):
        # The total number of time steps
        T = len(x)
        layers = []
        prev_s = np.zeros(self.hidden_dim)
        # For each time step...
        for t in range(T):
            layer = RNNCell()
            inputs = np.zeros(self.word_dim)
            inputs[x[t]] = 1
            layer.forward(inputs, prev_s, self.Wxh, self.Whh, self.Why)
            prev_s = layer.s
            layers.append(layer)
        return layers

    def predict(self, x):
        output = ActivationFunctions.Softmax()
        layers = self.forward_propagation(x)
        return [np.argmax(output.predict(layer.pred)) for layer in layers]

    def calculate_loss(self, x, y):
        '''
        하나의 sequence에 대해 loss를 계산하는 함수.
        '''
        assert len(x) == len(y)
        output = Softmax()
        layers = self.forward_propagation(x)
        loss = 0.0
        for i, layer in enumerate(layers):
            loss += output.loss(layer.pred, y[i])
        return loss / float(len(y))

    def calculate_total_loss(self, X, Y):
        '''
        sequence자체가 여러개 일 때 loss를 계산한다.
        '''
        loss = 0.0
        for i in range(len(Y)):
            loss += self.calculate_loss(X[i], Y[i])
        return loss / float(len(Y))

    def bptt(self, x, y):
        assert len(x) == len(y)
        output = Softmax()
        layers = self.forward_propagation(x)
        dWxh = np.zeros(self.Wxh.shape)
        dWhh = np.zeros(self.Whh.shape)
        dWhy = np.zeros(self.Why.shape)

        T = len(layers)
        prev_s_t = np.zeros(self.hidden_dim)
        diff_s = np.zeros(self.hidden_dim)
        for t in range(0, T):
            dPred = output.diff(layers[t].pred, y[t])
            inputs = np.zeros(self.word_dim)
            inputs[x[t]] = 1
            dprev_s, dWxh_t, dWhh_t, dWhy_t = layers[t].backward(inputs,
                                                                 prev_s_t,
                                                                 self.Wxh,
                                                                 self.Whh,
                                                                 self.Why,
                                                                 diff_s,
                                                                 dPred)
            prev_s_t = layers[t].s
            
            dPred = np.zeros(self.word_dim)  # prediction에 대한 gradient를 0으로 만들고 정해진 갯수만큼만 backpropagation해본다.
            for i in range(t-1, max(-1, t-self.bptt_truncate-1), -1):
                inputs = np.zeros(self.word_dim)
                inputs[x[i]] = 1
                prev_s_i = np.zeros(self.hidden_dim) if i == 0 else layers[i-1].s
                dprev_s, dWxh_i, dWhh_i, dWhy_i = layers[i].backward(inputs,
                                                                     prev_s_i,
                                                                     self.Wxh,
                                                                     self.Whh,
                                                                     self.Why,
                                                                     dprev_s,
                                                                     dPred)
                dWxh_t += dWxh_i
                dWhh_t += dWhh_i
            dWxh += dWxh_t
            dWhh += dWhh_t
            dWhy += dWhy_t

        return (dWxh, dWhh, dWhy)

    def sgd_step(self, x, y, learning_rate):
        dWxh, dWhh, dWhy = self.bptt(x, y)
        self.Wxh -= learning_rate * dWxh
        self.Whh -= learning_rate * dWhh
        self.Why -= learning_rate * dWhy

    def train(self, X, Y, learning_rate=0.005, nepoch=100, evaluate_loss_after=5):
        num_examples_seen = 0
        losses = []
        for epoch in range(nepoch):
            if (epoch % evaluate_loss_after == 0):
                loss = self.calculate_total_loss(X, Y)
                losses.append((num_examples_seen, loss))
                time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print("%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, loss))
                # Adjust the learning rate if loss increases
                if len(losses) > 1 and losses[-1][1] > losses[-2][1]:
                    learning_rate = learning_rate * 0.5
                    print("Setting learning rate to %f" % learning_rate)
                sys.stdout.flush()
            # For each training example...
            for i in range(len(Y)):
                self.sgd_step(X[i], Y[i], learning_rate)
                num_examples_seen += 1
        return losses 
