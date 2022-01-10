import numpy as np
import cvxopt
import matplotlib.pyplot as plt


def load_data():
    from sklearn.model_selection import train_test_split

    X = np.genfromtxt('mush_features.csv')
    Y = np.genfromtxt('mush_labels.csv')

    train_set_x, test_set_x, train_set_y, test_set_y = train_test_split(X, Y, test_size=0.33, random_state=42)

    train_set_x = train_set_x[:300].astype(float)
    train_set_y = train_set_y[:300].astype(float)

    test_set_x = test_set_x[:100].astype(float)
    test_set_y = test_set_y[:100].astype(float)

    x_test = train_set_x[:5]
    y_test = train_set_y[:5]

    train_set_x = train_set_x.reshape(train_set_x.shape[0], -1).T
    test_set_x = test_set_x.reshape(test_set_x.shape[0], -1).T

    train_set_y = train_set_y.reshape((1, train_set_y.shape[0]))
    test_set_y = test_set_y.reshape((1, test_set_y.shape[0]))

    x_test = x_test.reshape(x_test.shape[0], -1).T
    y_test = y_test.reshape((1, y_test.shape[0]))

    return train_set_x, test_set_x, train_set_y, test_set_y, x_test, y_test


class Kernel(object):
    def linear():
        return lambda x, y: np.dot(x, y)

    def polynomial(coef, power):
        return lambda x, y: (np.dot(x, y)+coef)**power

    def rbf(gamma):
        return lambda x, y: np.exp(-gamma*(np.linalg.norm(x.T-y, axis=0))**2)


class SVM(object):
    """
    The Support Vector Machines classifier
    """

    def __init__(self, C=1, kernel=Kernel.linear()):
        self.C = C
        self.kernel = kernel
        self.non_zero_multipliers = None
        self.support_vectors = None
        self.support_labels = None
        self.b = None

    def _kernel_matrix(self, X):
        """
        Computes kernel matrix applying kernel function pairwise for each sample
        """
        # Get number of samples
        n_samples = X.shape[1]

        # Calculate kernels pairwise and fill kernels matrix
        K = self.kernel(X.T, X)

        # Return kernel matrix
        return K

    def _compute_lagrange_multipliers(self, X, Y):
        """
        Solves the quadratic optimization problem and calculates lagrange multipliers
        """
        # Get number of samples
        n_samples = X.shape[1]

        K = self._kernel_matrix(X)

        # Create create quadratic term P based on Kernel matrix
        P = cvxopt.matrix(Y*np.transpose(Y)*K)

        # Create linear term q
        q = cvxopt.matrix(-np.ones((n_samples, 1)))

        # Create G, h
        if not self.C:
            G = cvxopt.matrix(np.identity(n_samples) * -1)
            h = cvxopt.matrix(np.zeros(n_samples))
        else:
            G_max = np.identity(n_samples) * -1
            G_min = np.identity(n_samples)

            G = cvxopt.matrix(np.vstack((G_max, G_min)))

            h_max = cvxopt.matrix(np.zeros(n_samples))
            h_min = cvxopt.matrix(np.ones(n_samples) * self.C)

            h = cvxopt.matrix(np.vstack((h_max, h_min)))

        # Create A, b
        A = cvxopt.matrix(Y.reshape(1, X.shape[1]))
        b = cvxopt.matrix(np.zeros(1))

        # Solve the quadratic optimization problem using cvxopt
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        # Extract flat array of lagrange multipliers
        lagrange_multipliers = np.ravel(solution['x'])

        return lagrange_multipliers

    def _get_support_vectors(self, lagrange_multipliers, X, Y):
        """
        Extracts the samples that will act as support vectors and corresponding labels
        """
        # Get indexes of non-zero lagrange multipiers
        idx = lagrange_multipliers > 1e-7

        # Get the corresponding lagrange multipliers
        non_zero_multipliers = lagrange_multipliers[idx]

        # Get the samples that will act as support vectors

        support_vectors = X[:, idx]

        # Get the corresponding labels
        support_labels = Y[:, idx]

        return non_zero_multipliers, support_vectors, support_labels

    def fit(self, X, Y):

        # Solve the quadratic optimization problem and get lagrange multipliers
        lagrange_multipliers = self._compute_lagrange_multipliers(X, Y)

        # Extract support vectors and non zero lagrange multipliers
        self.non_zero_multipliers, self.support_vectors, self.support_labels = self._get_support_vectors(lagrange_multipliers, X, Y)

        # Calculate b using first support vector
        self.b = self.support_labels[:, 0] - np.sum(self.non_zero_multipliers*self.support_labels*self._kernel_matrix(self.support_vectors)[0])

    def predict(self, X):
        """
        Predict function
        """
        n_samples = X.shape[1]

        predictions = np.sign(self.b + np.sum(self.non_zero_multipliers*self.support_labels*self.kernel(X.T, self.support_vectors), axis=1))
        predictions = np.reshape(predictions, (1, -1))

        return predictions


def plot(model, X, Y, grid_size):

    import matplotlib.cm as cm
    import itertools

    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_size),
        np.linspace(y_min, y_max, grid_size),
        indexing='ij'
    )

    def flatten(m): return np.array(m).reshape(-1,)

    result = []

    model.fit(X, Y)

    for (i, j) in itertools.product(range(grid_size), range(grid_size)):
        point = np.array([[xx[i, j]], [yy[i, j]]])
        result.append(model.predict(point)[0, 0])

    print(np.array(result).shape)
    print(xx.shape)

    Z = np.array(result).reshape(xx.shape)

    plt.contourf(
        xx, yy, Z,
        cmap=cm.Paired,
        levels=[-0.01, 0.01],
        extend='both',
        alpha=0.7
    )

    plt.scatter(
        flatten(X[0, :]),
        flatten(X[1, :]),
        c=flatten(Y),
        cmap=cm.Paired,
    )

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.show()


def accuracy(predictions, labels):
    return np.sum(predictions == labels, axis=1) / float(labels.shape[1])


train_set_x, test_set_x, train_set_y, test_set_y, x_test, y_test = load_data()
# Count examples classes
plt.figure(figsize=(4, 3))
plt.hist(train_set_y.T)
plt.xlabel("Class")
plt.ylabel("Count")
plt.tight_layout()
plt.show()


clf = SVM(C=1, kernel=Kernel.linear())
clf.fit(train_set_x, train_set_y)
y_pred = clf.predict(test_set_x)
accuracy(y_pred, test_set_y)

# Example results linear, polynomial and rbf models
samples = np.random.normal(size=200).reshape(2, 100)
labels = (2 * (samples.sum(axis=0) > 0) - 1.0).reshape(1, 100)
clf_lin = SVM(C=1, kernel=Kernel.linear())
plot(clf_lin, samples, labels, 200)
clf_polynomial = SVM(C=1, kernel=Kernel.polynomial(1, 3))
plot(clf_polynomial, samples, labels, 200)
clf_rbf = SVM(C=1, kernel=Kernel.rbf(0.03))
plot(clf_rbf, samples, labels, 200)
