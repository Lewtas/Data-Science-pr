import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d

def normalize(X):
    """
    Normalise data before processing
    """

    mu=np.mean(X,axis=0)
    se=np.std(X,axis=0)
    X_norm = (X-mu)/se
    norm_parameters=np.array([mu])
    norm_parameters = np.append(norm_parameters,se).reshape(2,-1)

    return X_norm, norm_parameters

class Eigendecomposition():

    def covariance(self, X):
        """
        Calculates eigenvectors and eigenvalues of covariance matrix
        """

        temp = np.dot((X).T,(X))/(X.shape[0]-1)
        e_val, e_vect=np.linalg.eig(temp)
        return e_val, e_vect

    def correlation(self, X):
        """
        Calculates eigenvectors and eigenvalues of correlation matrix
        """

        temp=np.dot((X).T,(X))/X.shape[0]
        e_val, e_vect=np.linalg.eig(temp)
        return e_val, e_vect

    def svd(self, X):
        """
        Calculates eigenvectors and eigenvalues by svd
        """

        u,s,v=np.linalg.svd(X.T)
        e_val=s**2/(X.shape[0]-1)
        e_vect=u
        return e_val, e_vect

class PCA():

    def __init__(self, X, n, eigendecomposition):
        self.X = X
        self.n = n
        self.eigendecomposition = eigendecomposition
        self.X_norm = None
        self.norm_params = None


    def transform(self):
        """
        Transforms the samples into the new subspace
        """

        self.X_norm , self.norm_params = normalize(self.X)
        e_val, e_vec = self.eigendecomposition(self.X_norm)
        e_vec=e_vec[:,np.flip(np.argsort(e_val))]
        e_val=np.sort(e_val)
        matrix_w=e_vec[:,:self.n]
        transformed = np.dot(self.X_norm,matrix_w)

        return transformed, matrix_w


    def restore(self):
        """
        Restores "original" values
        """


        a,b=self.transform()
        a=(np.dot(a,b.T))
        return a*self.norm_params[1]+self.norm_params[0]


def ImageExample():
    img = Image.open('little_girl.jpg')
    img = img.convert('L', colors=256)
    img = np.array(img, dtype=np.uint8)
    pca_img = PCA(img, 100, Eigendecomposition().svd)
    reduced_img, reduced_eigenvects_img = pca_img.transform()
    restored_img = pca_img.restore()
    imgplot = plt.imshow(img, cmap='gray')
    print('Initial:')
    plt.show()
    imgplot = plt.imshow(restored_img, cmap='gray')
    print('Restored:')
    plt.show()

#Basic example of transition from 3D to 2D space
#Step 1 create array of points and visualize space features
X = np.asarray([[12, 15, 20, 24, 27, 30, 63, 8, 67, 43, 11, 15, 67],
                [34, 31, 29, 88, 76, 80, 89, 53, 48, 66, 45, 50, 85],
                [45, 50, 43, 60, 65, 59, 89, 53, 43, 31, 33, 40, 80]]).T
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')
plt.rcParams['legend.fontsize'] = 10
ax.plot(X.T[0,:], X.T[1,:], X.T[2,:], 'o', markersize=8, color='blue', alpha=0.5, label='Initial')

ax.legend(loc='upper right')

plt.show()

#Use alghoritm with correlation matrix, svd or covaration and visualize new space
pca = PCA(X, 2, Eigendecomposition().correlation)

reduced_x, reduced_eigenvects = pca.transform()
print('Reduced input matrix:')
print(reduced_x[:5])

plt.plot(reduced_x.T[0,:], reduced_x.T[1,:], 'o', markersize=7, color='blue', alpha=0.5, label='Reduced')

plt.xlabel('x_values')
plt.ylabel('y_values')
plt.legend()

plt.show()

# Restore data and visualize restored space
new_x = pca.restore()
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(X.T[0,:], X.T[1,:], X.T[2,:], 'o', markersize=8, color='blue', alpha=0.5, label='Initial')
ax.plot(new_x.T[0,:], new_x.T[1,:], new_x.T[2,:], '^', markersize=8, alpha=0.5, color='red', label='Restored')

ax.legend(loc='upper right')

plt.show()

# And a small example of how you can
# compress a photo with the slightest loss of quality
# Now used svd
ImageExample()
