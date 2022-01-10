import math
import numpy as np
import matplotlib.pyplot as plt


def load_data():
    from sklearn.model_selection import train_test_split

    data = np.genfromtxt('time_temp_2016.tsv', delimiter='\t')

    x = data[:, 0]
    x = x.reshape((x.shape[0], 1))
    y = data[:, 1]

    train_set_x, test_set_x, train_set_y, test_set_y = train_test_split(x, y, test_size=0.33, random_state=42)

    train_set_y = train_set_y.reshape((1, train_set_y.shape[0]))
    test_set_y = test_set_y.reshape((1, test_set_y.shape[0]))

    return train_set_x.T, test_set_x.T, train_set_y, test_set_y, x.T


def polynomial_features(X, degree):

    from itertools import combinations_with_replacement

    n_features, n_samples = np.shape(X)

    def index_combinations():
        combs = [combinations_with_replacement(range(n_features), i) for i in range(0, degree + 1)]
        flat_combs = [item for sublist in combs for item in sublist]
        return flat_combs

    combinations = index_combinations()

    n_output_features = len(combinations)

    X_new = np.empty((n_output_features, n_samples))

    for i, index_combs in enumerate(combinations):
        X_new[i, :] = np.prod(X[index_combs, :], axis=0)
    return X_new


def mean_squared_error(y_true, y_pred):
    """ Returns the mean squared error between y_true and y_pred
    """
    mse = np.sum((y_pred-y_true)**2)/y_true.size

    return mse


class l2_regularization():
    """ Regularization for Ridge Regression """

    def __init__(self, alpha):
        """ Set alpha """
        self.alpha = alpha

    def __call__(self, w):
        """
        Computes l2 regularization term
        """
        term = 0.5*self.alpha*np.sum((w**2))
        return term

    def grad(self, w):
        """
        Computes derivative of l2 regularization term
        """
        derivative = self.alpha*w

        return derivative


class PolynomialRidgeRegression(object):

    def __init__(self, degree, reg_factor, n_iterations=3000, learning_rate=0.01, print_error=False):
        self.degree = degree
        self.regularization = l2_regularization(alpha=reg_factor)
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.print_error = print_error

    def initialize_with_zeros(self, n_features):
        """
        This function creates a vector of zeros of shape (n_features, 1)
        """
        self.w = np.zeros((n_features, 1))

    def fit(self, X, Y):
        # Generate polynomial features
        X = polynomial_features(X, self.degree)

        # Create array
        self.initialize_with_zeros(n_features=X.shape[0])

        # Do gradient descent for n_iterations
        for i in range(self.n_iterations):
            # Calculate prediction
            H = np.dot(self.w.T, X)

            # Gradient of l2 loss w.r.t w
            grad_w = np.dot(X, (H-Y).T) + self.regularization.grad(self.w)

            # Update the weights
            self.w = self.w-self.learning_rate*grad_w

            if self.print_error and i % 1000 == 0:
                # Calculate l2 loss
                mse = mean_squared_error(Y, H)
                print("MSE after iteration %i: %f" % (i, mse))

    def predict(self, X):
        # Generate polynomial features
        X = polynomial_features(X, self.degree)

        # Calculate prediction
        y_pred = np.dot(self.w.T, X)

        return y_pred


train_set_x, test_set_x, train_set_y, test_set_y, full_feature_set_for_plot = load_data()

poly_degree = 15
learning_rate = 0.001
n_iterations = 10000
reg_factor = 0.1

cmap = plt.get_cmap('viridis')

# Plot the results
m1 = plt.scatter(366 * train_set_x, train_set_y, color=cmap(0.9), s=10)
m2 = plt.scatter(366 * test_set_x, test_set_y, color=cmap(0.5), s=10)
plt.xlabel('Day')
plt.ylabel('Temperature in Celcius')
plt.legend((m1, m2), ("Training data", "Test data"), loc='lower right')
plt.show()

model = PolynomialRidgeRegression(
    degree=poly_degree,
    reg_factor=reg_factor,
    learning_rate=learning_rate,
    n_iterations=n_iterations,
    print_error=True
)

model.fit(train_set_x, train_set_y)

y_predictions = model.predict(test_set_x)

mse = mean_squared_error(test_set_y, y_predictions)

print("Mean squared error on test set: %s (given by reg. factor: %s)" % (mse, reg_factor))


cmap = plt.get_cmap('viridis')

# Predict for all points in set
y_val = model.predict(full_feature_set_for_plot)

# Plot the results
m1 = plt.scatter(366 * train_set_x, train_set_y, color=cmap(0.9), s=10)
m2 = plt.scatter(366 * test_set_x, test_set_y, color=cmap(0.5), s=10)
plt.plot(366 * full_feature_set_for_plot.T, y_val.T, color='black', linewidth=2, label="Prediction")
plt.suptitle("Polynomial Ridge Regression")
plt.title("MSE: %.2f" % mse, fontsize=10)
plt.xlabel('Day')
plt.ylabel('Temperature in Celcius')
plt.legend((m1, m2), ("Training data", "Test data"), loc='lower right')
plt.show()
