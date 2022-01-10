import numpy as np
import matplotlib.pyplot as plt


def load_data():
    from sklearn.model_selection import train_test_split

    data = np.genfromtxt('kangaroo.csv', delimiter=',')

    x = data[:, 0]
    y = data[:, 1]

    train_set_x, test_set_x, train_set_y, test_set_y = train_test_split(x, y, test_size=0.33, random_state=42)

    return train_set_x, test_set_x, train_set_y, test_set_y


def initialize_with_zeros():

    theta = 0
    b = 0

    assert(isinstance(theta, int))
    assert(isinstance(b, int))

    return theta, b


def propagate(theta, b, X, Y):

    m = X.shape[0]

    # FORWARD PROPAGATION (FROM X TO COST)
    H = theta*X + b        # compute activation
    cost = (1/(2*m)) * ((H-Y)**2).sum()    # compute cost

    # BACKWARD PROPAGATION (TO FIND GRAD)
    dt = (1/m)*X.dot((np.transpose(H-Y)))

    db = (1/m)*(H-Y).sum()

    assert(dt.dtype == float)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())

    grads = {"dt": dt,
             "db": db}

    return grads, cost


def optimize(theta, b, X, Y, num_iterations, learning_rate, print_cost=False):

    costs = []

    for i in range(num_iterations):

        # Cost and gradient calculation
        grads, cost = propagate(theta, b, X, Y)

        # Retrieve derivatives from grads
        dt = grads["dt"]
        db = grads["db"]

        # update rule
        theta = theta - learning_rate*dt
        b = b-learning_rate*db

        # Record the costs
        if i % 100 == 0:
            costs.append(cost)

        # Print the cost every 100 training iterations
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    params = {"theta": theta,
              "b": b}

    grads = {"dt": dt,
             "db": db}

    return params, grads, costs


def predict(theta, b, X):

    # Compute vector "Y_prediction" predicting the width of a kangoroo nasal
    Y_prediction = theta*X + b

    return Y_prediction


def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):

    # initialize parameters with zeros
    theta, b = initialize_with_zeros()

    # Gradient descent
    parameters, grads, costs = optimize(theta, b, X_train, Y_train, num_iterations, learning_rate, print_cost)

    # Retrieve parameters w and b from dictionary "parameters"
    theta = parameters["theta"]
    b = parameters["b"]

    # Predict test/train set examples
    Y_prediction_test = theta*X_test + b
    Y_prediction_train = theta*X_train + b

    # Print train/test Errors
    print("Train RMSE: {} ".format(np.sqrt(np.mean((Y_prediction_train - Y_train) ** 2))))
    print("Test RMSE: {} ".format(np.sqrt(np.mean((Y_prediction_test - Y_test) ** 2))))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "theta": theta,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}

    return d


train_set_x, test_set_x, train_set_y, test_set_y = load_data()

############################

m_train = train_set_x.shape[0]
m_test = test_set_x.shape[0]


############################
mean = np.concatenate([train_set_x, test_set_x]).mean()

std = np.concatenate([train_set_x, test_set_x]).std()

train_set_x = (train_set_x - mean) / std
test_set_x = (test_set_x - mean) / std

d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=500, learning_rate=0.05, print_cost=True)

############################

plt.figure(figsize=(6, 4))
plt.title("Training set")

plt.scatter(train_set_x, train_set_y)
x = np.array([min(train_set_x), max(train_set_x)])
theta = d["theta"]
b = d["b"]
y = theta * x + b
plt.plot(x, y)
plt.axis("tight")
plt.xlabel("Length")
plt.ylabel("Width")
plt.tight_layout()


# Test set
plt.figure(figsize=(6, 4))
plt.title("Test set")

plt.scatter(test_set_x, test_set_y)
x = np.array([min(test_set_x), max(test_set_x)])
theta = d["theta"]
b = d["b"]
y = theta * x + b
plt.plot(x, y)
plt.axis("tight")
plt.xlabel("Length")
plt.ylabel("Width")
plt.tight_layout()
plt.show()
