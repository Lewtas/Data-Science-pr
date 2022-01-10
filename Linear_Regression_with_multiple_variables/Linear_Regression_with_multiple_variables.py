
import numpy as np
import matplotlib.pyplot as plt


def load_data():
    ''' load dataset from sklearn lib
        The Boston housing prices dataset'''

    from sklearn.datasets import load_boston
    from sklearn.model_selection import train_test_split

    boston = load_boston()

    train_set_x, test_set_x, train_set_y, test_set_y = train_test_split(boston.data, boston.target, test_size=0.33, random_state=42)

    train_set_y = train_set_y.reshape((1, train_set_y.shape[0]))
    test_set_y = test_set_y.reshape((1, test_set_y.shape[0]))

    return train_set_x.T, train_set_y, test_set_x.T, test_set_y, boston


def initialize_with_zeros(dim):
    """
    This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.
    """
    w = np.zeros((dim, 1))
    b = 0

    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))

    return w, b


def propagate(w, b, X, Y):
    """
    Implement the cost function and its gradient for the propagation explained above
    """

    m = X.shape[1]

    # FORWARD PROPAGATION (FROM X TO COST)

    H = w.T.dot(X) + b  # compute activation
    cost = (1/(2*m))*((H-Y).T**2).sum()  # compute cost

    # BACKWARD PROPAGATION (TO FIND GRAD)

    dw = (1/m)*X.dot((H-Y).T)

    db = (1/m)*(H-Y).sum()

    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())

    grads = {"dw": dw,
             "db": db}

    return grads, cost


def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    """
    This function optimizes w and b by running a gradient descent algorithm
    """

    costs = []

    for i in range(num_iterations):

        # Cost and gradient calculation
        grads, cost = propagate(w, b, X, Y)

        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]

        # update rule
        w = w - learning_rate*dw
        b = b - learning_rate*db

        # Record the costs
        if i % 100 == 0:
            costs.append(cost)

        # Print the cost every 100 training iterations
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    params = {"w": w,
              "b": b}

    grads = {"dw": dw,
             "db": db}

    return params, grads, costs


def predict(w, b, X):
    """
    Predict using learned linear regression parameters (w, b)
    """

    m = X.shape[1]

    # Compute vector "H"
    H = w.T.dot(X) + b
    assert(H.shape == (1, m))

    return H


def show_features(visualization_set):
    for index, feature_name in enumerate(visualization_set.feature_names):
        plt.figure(figsize=(6, 4))
        plt.scatter(visualization_set.data[:, index], visualization_set.target)
        plt.ylabel("Price", size=15)
        plt.xlabel(feature_name, size=15)
        plt.tight_layout()
    plt.show()


def model(X_train, Y_train, X_test, Y_test, num_iterations=3000, learning_rate=0.5, print_cost=False):
    """
    Builds the linear regression model by calling the function you've implemented previously
    """

    # initialize parameters with zeros
    w, b = initialize_with_zeros(X_train.shape[0])

    # Gradient descent
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)

    # Retrieve parameters w and b from dictionary "parameters"
    w = parameters["w"]
    b = parameters["b"]

    # Predict test/train set examples
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    # Print train/test Errors
    print("Train RMSE: {} ".format(np.sqrt(np.mean((Y_prediction_train - Y_train) ** 2))))
    print("Test RMSE: {} ".format(np.sqrt(np.mean((Y_prediction_test - Y_test) ** 2))))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}

    return d


train_set_x, train_set_y, test_set_x, test_set_y, visualization_set = load_data()
all_set_x = np.concatenate([train_set_x, test_set_x], axis=1)

mean = all_set_x.mean(axis=1, keepdims=True)
std = all_set_x.std(axis=1, keepdims=True)

train_set_x = (train_set_x - mean) / std
test_set_x = (test_set_x - mean) / std
d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=3000, learning_rate=0.05, print_cost=False)


plt.figure(figsize=(4, 3))
plt.title("Training set")
plt.scatter(train_set_y, d["Y_prediction_train"])
plt.plot([0, 50], [0, 50], "--k")
plt.axis("tight")
plt.xlabel("True price ($1000s)")
plt.ylabel("Predicted price ($1000s)")
plt.tight_layout()

# Test set
plt.figure(figsize=(4, 3))
plt.title("Test set")
plt.scatter(test_set_y, d["Y_prediction_test"])
plt.plot([0, 50], [0, 50], "--k")
plt.axis("tight")
plt.xlabel("True price ($1000s)")
plt.ylabel("Predicted price ($1000s)")
plt.tight_layout()
plt.show()

show_features(visualization_set)
