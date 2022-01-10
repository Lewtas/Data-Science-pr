import numpy as np
import matplotlib.pyplot as plt

def load_data():
    ''' Include dataset iris from sklearn'''
    from sklearn.model_selection import train_test_split
    from sklearn import datasets

    iris = datasets.load_iris()

    train_set_x, test_set_x, train_set_y, test_set_y = train_test_split(iris.data, iris.target, test_size=0.33, random_state=42)

    return train_set_x, test_set_x, train_set_y, test_set_y, iris

def euclidian_dist(x_known,x_unknown):
    """
    This function calculates euclidian distance between each pairs of known and unknown points
    """
    num_pred = x_unknown.shape[0]
    num_data = x_known.shape[0]


    dists = np.empty((num_pred,num_data))

    for i in range(num_pred):
        for j in range(num_data):
            # calculate euclidian distance here
            dists[i,j] = np.sqrt(np.sum((x_unknown[i]-x_known[j])**2))

    return dists

def k_nearest_labels(dists, y_known, k):
    """
    This function returns labels of k-nearest neighbours to each sample for unknown data.
    """

    num_pred = dists.shape[0]
    n_nearest = []

    for j in range(num_pred):
        dst = dists[j]

        # count k closest points
        t=k
        if(t>=dst.shape[0]):
            t=dst.shape[0]-1
        closest_y = y_known[np.argpartition(dst,t)[:k]]

        n_nearest.append(closest_y)
    return np.asarray(n_nearest)


class KNearest_Neighbours(object):

    def __init__(self, k):

        self.k = k
        self.test_set_x = None
        self.train_set_x = None
        self.train_set_y = None


    def fit(self, train_set_x, train_set_y):

        self.train_set_x=train_set_x
        self.train_set_y=train_set_y


    def predict(self, test_set_x):

        # Returns list of predicted labels for test set; type(prediction) -> list, len(prediction) = len(test_set_y)
        self.test_set_x=test_set_x

        return np.round((np.sum((k_nearest_labels(euclidian_dist(self.train_set_x,self.test_set_x),self.train_set_y,self.k)),axis=1)/self.k))



train_set_x, test_set_x, train_set_y, test_set_y, visualization_set = load_data()

plt.figure(figsize=(4, 3))
plt.hist(visualization_set.target)
plt.xlabel("Class")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

for index, feature_name in enumerate(visualization_set.feature_names):
    plt.figure(figsize=(4, 3))
    plt.scatter(visualization_set.data[:, index], visualization_set.target)
    plt.ylabel("Class", size=15)
    plt.xlabel(feature_name, size=15)
    plt.tight_layout()
plt.show()

k = 4
model = KNearest_Neighbours(k)
model.fit(train_set_x, train_set_y)
y_predictions = model.predict(test_set_x)
actual = list(test_set_y)
accuracy = (y_predictions == test_set_y).mean()
print(accuracy)
for index, feature_name in enumerate(visualization_set.feature_names):
    plt.figure(figsize=(4, 3))
    plt.scatter(test_set_x[:, index], test_set_y) # real labels
    plt.scatter(test_set_x[:, index], y_predictions) # predicted labels
    plt.ylabel("Class", size=15)
    plt.xlabel(feature_name, size=15)
    plt.tight_layout()
plt.show()
