from matplotlib import animation
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()




class KMeans(object):

    def __init__(self, X, k):
        self.X = X
        self.k = k

    def initialize_centroids(self):

        temp = self.X.copy()
        np.random.shuffle(temp)
        return temp[:self.k]

    def closest_centroid(self, centroids):

        a = np.array([np.sum((self.X-i)**2, axis=1) for i in centroids])
        return np.argmin(a, axis=0)

    def move_centroids(self, centroids):

        temp = self.closest_centroid(centroids)
        for i in range(self.k):
            S = self.X[temp == i]
            centroids[i] = np.sum(S, axis=0)/(S.shape[0])
        return centroids


    def final_centroids(self):


        centroids = self.initialize_centroids()
        temp_centroids = np.random.random(centroids.shape)
        while np.any(centroids != temp_centroids):
            temp_centroids = centroids.copy()
            centroids = self.move_centroids(centroids)

        temp = self.closest_centroid(centroids)
        clusters = [self.X[temp == i] for i in range(self.k)]

        return clusters, centroids


X = np.vstack(((np.random.randn(150, 2) + np.array([3, 0])),
               (np.random.randn(100, 2) + np.array([-3.5, 0.5])),
               (np.random.randn(100, 2) + np.array([-0.5, -2])),
               (np.random.randn(150, 2) + np.array([-2, -2.5])),
               (np.random.randn(150, 2) + np.array([-5.5, -3]))))



model = KMeans(X, 3)

centroids = model.initialize_centroids()
print('Random centroids:', centroids)

plt.scatter(X[:, 0], X[:, 1], s=30)
plt.scatter(centroids[:, 0], centroids[:, 1], s=600, marker='*', c='r')
ax = plt.gca()
plt.show()


closest = model.closest_centroid(centroids)
print('Closest centroids:', closest[:10])

plt.scatter(X[:, 0], X[:, 1], s=30, c=closest)
plt.scatter(centroids[:, 0], centroids[:, 1], s=600, marker='*', c='r')
ax = plt.gca()
plt.show()
next_centroids = model.move_centroids(centroids)
print('Next centroids:', next_centroids)

clusters, final_centrs = model.final_centroids()
print('Final centroids:', final_centrs)
print('Clusters points:', clusters[0][0], clusters[1][0], clusters[2][0])


fig = plt.figure()
ax = plt.gca()
centroids = model.initialize_centroids()
line, = ax.plot([], [], 'r*', markersize=15)





def mean_distances(k, X):


    a = np.zeros(k)
    for i in range(1, k+1):
      m = KMeans(X, i)
      clusters, ce = m.final_centroids()
      temp = m.closest_centroid(ce)

      for j in range(i):
        a[i-1] += np.sum((np.sum((X[temp == j]-ce[j])**2, axis=1)))/i

    return a


print('Mean distances: ', mean_distances(10, X))

k_clusters = range(1, 11)
distances = mean_distances(10, X)
plt.plot(k_clusters, distances)
plt.xlabel('k')
plt.ylabel('Mean distance')
plt.title('The Elbow Method showing the optimal k')
plt.show()
