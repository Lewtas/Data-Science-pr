import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageOps
import glob
import os
import re
import unarchive


def shuffle_data(features, labels):
    assert len(features) == len(labels)
    idx = np.random.permutation(len(features))
    return [a[idx] for a in [features, labels]]


def load_data():
    if(not os.path.exists('test')):
        unarchive_test()

    if(not os.path.exists('train')):
        unarchive_train()

    train_y = pd.read_csv('trainLabels.csv')
    filelist = glob.glob('train/*')

    train_x = np.array([np.asarray(Image.open(img)) for img in filelist])

    filelist = os.listdir('train/')
    for i in range(len(filelist)):
        filelist[i] = re.sub(r'.png', '', filelist[i])
        filelist[i] = int(filelist[i]) - 1
    train_y = train_y['label'][filelist]
    train_y = train_y.to_numpy()

    filelist = glob.glob('test/*')
    test_x = np.array([np.asarray(Image.open(img)) for img in filelist])

    return train_x, train_y, test_x


def NumToLable(y):
    global classes
    temp = np.empty(y.shape, dtype='object')

    for i in range(y.size):
        temp[i] = classes[y[i]]
    return temp


def one_hot(Y, n_classes):
    temp = np.zeros((Y.shape[0], n_classes.size))
    for i in range(temp.shape[0]):
        temp[i, np.where(n_classes == Y[i])] = 1
    return temp.T


class Sigmoid:
    def __call__(self, z):
        return 1/(1+np.exp(-z))

    def prime(self, z):
        return self.__call__(z)*(1-self.__call__(z))


def compute_cost(A2, Y):
    m = Y.shape[1]  # number of examples
    cost = np.sum(Y*np.log(A2) + (1-Y)*np.log(1-A2))
    cost /= -m
    return cost


def toLable(y_i):
    global classes
    return classes[y_i == 1]


def plot_digit(x_set, y_set, idx):
    img = x_set[idx].reshape(32, 32)
    plt.imshow(img, cmap='Greys',  interpolation='nearest')
    plt.title('true label: %s' % toLable(y_set[:, idx]))
    plt.show()


class Regularization:

    def __init__(self, lambda_1, lambda_2):
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2

    def l1(self, W1, W2, m):
        return (self.lambda_1/(m)) * (np.linalg.norm(W1, ord=1) + np.linalg.norm(W2, ord=1))

    def l1_grad(self, W1, W2, m):
        temp1 = W1 > 0
        temp1.dtype = 'int8'
        temp1[temp1 < 1] = -1
        temp2 = W2 > 0
        temp2.dtype = 'int8'
        temp2[temp2 < 1] = -1
        return {'dW1': self.lambda_1/(m)*temp1, 'dW2': self.lambda_1/(m)*temp2}

    def l2(self, W1, W2, m):
        return (self.lambda_2/(2*m)) * (np.linalg.norm(W1**2, ord=1) + np.linalg.norm(W2**2, ord=1))

    def l2_grad(self, W1, W2, m):
        return {'dW1': self.lambda_2/(m) * W1, 'dW2': self.lambda_2/(m) * W2}


class NeuralNetwork:

    def __init__(self, n_features, n_hidden_units, n_classes, learning_rate, reg=Regularization(0.05, 0.1), sigm=Sigmoid()):
        self.n_features = n_features
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.n_hidden_units = n_hidden_units
        self.reg = reg
        self.sigm = sigm
        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None
        self.initialize_parameters()

    def initialize_parameters(self):
        mu, sigma = 0, 0.01
        self.W1 = np.random.normal(mu, sigma, size=(self.n_hidden_units, self.n_features))
        self.W2 = np.random.normal(mu, sigma, size=(self.n_classes, self.n_hidden_units))
        self.b1 = np.zeros((self.n_hidden_units, 1))
        self.b2 = np.zeros((self.n_classes, 1))

    def forward_propagation(self, X):
        Z1 = np.dot(self.W1, X)+self.b1
        A1 = self.sigm(Z1)
        Z2 = np.dot(self.W2, A1)+self.b2
        A2 = self.sigm(Z2)
        return {
            'Z1': Z1,
            'A1': A1,
            'Z2': Z2,
            'A2': A2}

    def backward_propagation(self, X, Y, cache):

        m = X.shape[1]
        self.A1 = cache['A1']
        self.A2 = cache['A2']
        l1 = self.reg.l1_grad(self.W1, self.W2, m)
        l2 = self.reg.l2_grad(self.W1, self.W2, m)
        dW1 = ((1/m)*np.dot(np.dot(self.W2.T, (self.A2-Y))*self.sigm.prime(np.dot(self.W1, X)+self.b1), X.T)+l1['dW1']+l2['dW1'])
        dW2 = ((1/m)*np.dot((self.A2-Y), self.A1.T)+l1['dW2']+l2['dW2'])
        db1 = ((1/m)*np.sum(np.dot(self.W2.T, (self.A2-Y))*self.sigm.prime(np.dot(self.W1, X)+self.b1), axis=1, keepdims=True))
        db2 = ((1/m)*np.sum((self.A2-Y), axis=1, keepdims=True))

        return {
            'dW1': dW1,
            'db1': db1,
            'dW2': dW2,
            'db2': db2}

    def update_parameters(self, grads):

        self.dW1 = grads['dW1']
        self.dW2 = grads['dW2']

        self.db1 = grads['db1']
        self.db2 = grads['db2']

        self.W1 = self.W1-self.learning_rate*self.dW1

        self.W2 = self.W2-self.learning_rate*self.dW2

        self.b1 = self.b1-(self.learning_rate*self.db1)

        self.b2 = self.b2-(self.learning_rate*self.db2)


class NNClassifier:

    def __init__(self, model, epochs=1000):
        self.model = model
        self.epochs = epochs
        self._cost = []  # Collect values of cost function after each epoch to build graph later

    def fit(self, X, Y):

        Y = one_hot(Y, np.unique(Y))

        self.model.initialize_parameters()
        i = 0
        while i < self.epochs:
            print(i)
            self.model.update_parameters(self.model.backward_propagation(X, Y, self.model.forward_propagation(X)))
            self._cost.append(compute_cost(self.model.A2, Y))
            i += 1

    def predict(self, X):
        cache = self.model.forward_propagation(X)
        return np.argmax(cache['A2'], axis=0).T


def accuracy(pred, labels):
    return (np.sum(NumToLable(pred) == labels) / float(labels.size))


def plot_error(model, epochs):
    plt.plot(range(len(model._cost)), model._cost)
    plt.ylim([0, epochs])
    plt.ylabel('Error')
    plt.xlabel('Epochs')
    plt.show()


train_set_x, train_set_y, test_set_x = load_data()
print(train_set_x.shape)
classes = np.unique(train_set_y)


NN = NeuralNetwork(train_set_x.shape[1], 40, classes.size, 0.001)
classifier = NNClassifier(NN, 5000)

classifier.fit(train_set_x.T, train_set_y)
plot_error(classifier, 100)

pred_train = classifier.predict(train_set_x.T)
print(pred_train, pred_train.shape)
print('train set accuracy: ', accuracy(pred_train, train_set_y))

pred_train = classifier.predict(test_set_x.T)

submission = pd.DataFrame(columns=['id', 'label'], dtype=str)
submission['label'] = [classes[int(i)] for i in pred_train]
filelist = os.listdir('test/')
submission['id'] = [(''.join(filter(str.isdigit, name))) for name in filelist]
