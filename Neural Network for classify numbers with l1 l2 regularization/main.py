import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style='whitegrid', palette='muted', font_scale=1.5)


def read_mnist(images_path, labels_path):
    import struct
    import os
    with open(labels_path, 'rb') as p:
        magic, n = struct.unpack('>II', p.read(8))
        labels = np.fromfile(p, dtype=np.uint8)
    with open(images_path, 'rb') as p:
        magic, num, rows, cols = struct.unpack(">IIII", p.read(16))
        images = np.fromfile(p, dtype=np.uint8).reshape(len(labels), 784)

    return images, labels

# Shuffle dataset


def shuffle_data(features, labels):
    assert len(features) == len(labels)

    idx = np.random.permutation(len(features))
    return [a[idx] for a in [features, labels]]

# Loading data


def plot_digit(x_set, y_set, idx):
    img = x_set.T[idx].reshape(28, 28)
    plt.imshow(img, cmap='Greys',  interpolation='nearest')
    plt.title('true label: %d' % y_set.T[idx])
    plt.show()


def load_data():
    X, y = read_mnist('samples/train-images-idx3-ubyte', 'samples/train-labels-idx1-ubyte')
    X, y = shuffle_data(X, y)
    train_set_x, train_set_y = X[:5000], y[:5000]
    test_set_x, test_set_y = X[5000:], y[5000:]

    test_set_x = test_set_x.reshape(test_set_x.shape[0], -1).T
    train_set_x = train_set_x.reshape(train_set_x.shape[0], -1).T
    train_set_y = train_set_y.reshape((1, train_set_y.shape[0]))
    test_set_y = test_set_y.reshape((1, test_set_y.shape[0]))

    return train_set_x, test_set_x, train_set_y, test_set_y


class Sigmoid:
    def __call__(self, z):
        """
        Compute the sigmoid of z
        """
        return 1/(1+np.exp(-z))

    def prime(self, z):
        """
        Compute the derivative of sigmoid of z
        """
        return self.__call__(z)*(1-self.__call__(z))


def one_hot(Y, n_classes):
    """
    Encode labels into a one-hot representation
    """
    temp = np.zeros((Y.shape[1], n_classes))
    Y = np.reshape(Y, -1)
    for i in range(temp.shape[0]):
      temp[i][Y[i]] = 1
    return temp.T


def compute_cost(A2, Y):
    """
    Computes the cross-entropy cost given in equation (4)
    """

    m = Y.shape[1]  # number of examples

    # Compute the cross-entropy cost

    cost = np.sum(Y*np.log(A2) + (1-Y)*np.log(1-A2))
    cost /= -m

    return cost


class Regularization:
    """
    Regularization class
    """

    def __init__(self, lambda_1, lambda_2):
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2

    def l1(self, W1, W2, m):
        """
        Compute l1 regularization part
        """
        return (self.lambda_1/(m)) * (np.linalg.norm(W1, ord=1) + np.linalg.norm(W2, ord=1))

    def l1_grad(self, W1, W2, m):
        """
        Compute l1 regularization term
        """
        temp1 = W1 > 0

        temp1.dtype = 'int8'
        temp1[temp1 < 1] = -1
        temp2 = W2 > 0
        temp2.dtype = 'int8'
        temp2[temp2 < 1] = -1
        return {'dW1': self.lambda_1/(m)*temp1, 'dW2': self.lambda_1/(m)*temp2}

    def l2(self, W1, W2, m):
        """
        Compute l2 regularization term
        """
        return (self.lambda_2/(2*m)) * (np.linalg.norm(W1**2, ord=1) + np.linalg.norm(W2**2, ord=1))

    def l2_grad(self, W1, W2, m):
        """
        Compute l2 regularization term
        """
        return {'dW1': self.lambda_2/(m) * W1, 'dW2': self.lambda_2/(m) * W2}


class NeuralNetwork:

    def __init__(self, n_features, n_hidden_units, n_classes, learning_rate, reg=Regularization(0.1, 0.2), sigm=Sigmoid()):
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

        # Implement Forward Propagation to calculate A2 (probabilities)
        Z1 = np.dot(self.W1, X)+self.b1
        A1 = self.sigm(Z1)
        Z2 = np.dot(self.W2, A1)+self.b2
        A2 = self.sigm(Z2)

        return {
            'Z1': Z1,
            'A1': A1,
            'Z2': Z2,
            'A2': A2
        }

    def backward_propagation(self, X, Y, cache):
        m = X.shape[1]

        # Retrieve A1 and A2 from dictionary "cache".
        self.A1 = cache['A1']
        self.A2 = cache['A2']

        # Calculate gradients for L1, L2 parts using attribute instance of Regularization class
        l1 = self.reg.l1_grad(self.W1, self.W2, m)
        l2 = self.reg.l2_grad(self.W1, self.W2, m)

        # Backward propagation: calculate dW1, db1, dW2, db2 (using obtained L1, L2 gradients)
        dW1 = ((1/m)*np.dot(np.dot(self.W2.T, (self.A2-Y))*self.sigm.prime(np.dot(self.W1, X)+self.b1), X.T)+l1['dW1']+l2['dW1'])
        dW2 = ((1/m)*np.dot((self.A2-Y), self.A1.T)+l1['dW2']+l2['dW2'])
        db1 = ((1/m)*np.sum(np.dot(self.W2.T, (self.A2-Y))*self.sigm.prime(np.dot(self.W1, X)+self.b1), axis=1, keepdims=True))
        db2 = ((1/m)*np.sum((self.A2-Y), axis=1, keepdims=True))

        return {
            'dW1': dW1,
            'db1': db1,
            'dW2': dW2,
            'db2': db2
        }

    def update_parameters(self, grads):
        """
        Updates parameters using the gradient descent update rule
        """
        # Retrieve each gradient from the dictionary "grads"

        self.dW1 = grads['dW1']
        self.dW2 = grads['dW2']

        self.db1 = grads['db1']
        self.db2 = grads['db2']

        # Update each parameter
        self.W1 = self.W1-self.learning_rate*self.dW1

        self.W2 = self.W2-self.learning_rate*self.dW2

        self.b1 = self.b1-(self.learning_rate*self.db1)

        self.b2 = self.b2-(self.learning_rate*self.db2)


class NNClassifier:
    """
    NNClassifier class
    """

    def __init__(self, model, epochs=1000):
        self.model = model
        self.epochs = epochs
        self._cost = []  # Collect values of cost function after each epoch to build graph later

    def fit(self, X, Y):
        """
        Learn weights and errors from training data
        """

        Y = one_hot(Y, np.unique(Y).size)

        self.model.initialize_parameters()
        i = 0
        while i < self.epochs:
            print(i)
            self.model.update_parameters(self.model.backward_propagation(X, Y, self.model.forward_propagation(X)))
            self._cost.append(compute_cost(self.model.A2, Y))
            i += 1

    def predict(self, X):
        """
        Generate array of predicted labels for the input dataset
        """

        cache = self.model.forward_propagation(X)

        return np.argmax(cache['A2'], axis=0).T


def accuracy(pred, labels):
    return (np.sum(pred == labels, axis=1) / float(labels.shape[1]))[0]


def plot_error(model, epochs):
    plt.plot(range(len(model._cost)), model._cost)
    plt.ylim([0, epochs])
    plt.ylabel('Error')
    plt.xlabel('Epochs')
    plt.show()


train_set_x, test_set_x, train_set_y, test_set_y = load_data()
NN = NeuralNetwork(784, 30, 10, 0.01)
classifier = NNClassifier(NN, 5000)
classifier.fit(train_set_x, train_set_y)
plot_error(classifier, 10)
pred_train = classifier.predict(train_set_x)
pred_test = classifier.predict(test_set_x)

print('train set accuracy: ', accuracy(pred_train, train_set_y))
print('test set accuracy: ', accuracy(pred_test, test_set_y))
plot_digit(test_set_x, test_set_y, idx=6)
pred_single = classifier.predict(test_set_x.T[6].reshape(784, 1))
print("The digit is " + str(pred_single[0]))
plot_digit(test_set_x, test_set_y, idx=90)
pred_single = classifier.predict(test_set_x.T[90].reshape(784, 1))
print("The digit is " + str(pred_single[0]))
