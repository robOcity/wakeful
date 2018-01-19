# Cite: Galvanize DSI clustering code 2017-12-07.
# note still has intermittent failure, unable to do > 6 clusters occasionally
import numpy as np
import random
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

class KMeans(object):
    '''
    K-Means clustering
    ----------
    n_clusters : int, optional, default: 8
        The number of clusters to form as well as the number of
        centroids to generate.
    init : {'random', 'random_initialization', 'k-means++'}
        Method for initialization, defaults to 'k-means++':
        'k-means++' : selects initial cluster centers for k-mean
        clustering in a smart way to speed up convergence. See section
        Notes in k_init for more details.
        'random': choose k observations (rows) at random from data for
        the initial centroids.
        If an ndarray is passed, it should be of shape (n_clusters, n_features)
        and gives the initial centers.
    n_init : int, default: 10
        Number of time the k-means algorithm will be run with different
        centroid seeds. The final results will be the best output of
        n_init consecutive runs in terms of inertia.
    max_iter : int, default: 1000
        Maximum number of iterations of the k-means algorithm for a
        single run.
    tolerance : int, default : .00001
    Attributes
    ----------
    cluster_centers_ : array, [n_clusters, n_features]
        Coordinates of cluster centers
    labels_ :
        Labels of each point
    '''

    def __init__(self, n_clusters=8, init='random', n_init=10,
                 max_iter=300, tolerance = 1e-4, verbose = False):

        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.n_init = n_init
        self.verbose = verbose
        self.centroids_ = None
        self.labels_ = None

    def _initialize_centroids(self, X):
        '''
        Parameters
        ----------
        X : array-like or sparse matrix, shape=(n_samples, n_features)
            Data points to take random selection from for initial centroids
        You should code the simplest case of random selection of k centroids from data
        OPTIONAL: code up random_initialization and/or k-means++ initialization here also
        '''
        randinds = np.random.choice(np.arange(X.shape[0]), self.n_clusters)
        self.centroids_ =  X[randinds]

    def _assign_clusters(self, X):
        '''
        computes euclidean distance from each point to each centroid and
        assigns point to closest centroid)
        assigns self.labels_
        Parameters
        ----------
        X : array-like or sparse matrix, shape=(n_samples, n_features)
            Data points to assign to clusters based on distance metric
        '''
        labels = np.zeros(X.shape[0])
        for i, row in enumerate(X):
            # finds centroid with minimum euclidean distance
            min_dist = np.argmin([np.linalg.norm((row - c)) for c in self.centroids_])
            labels[i] = min_dist
        self.labels_ = labels.astype('int')

    def _compute_centroids(self, X):
        '''
        compute the centroids for the datapoints in X from the current values
        of self.labels_
        assigns self.centroids_
        Parameters
        ----------
        X : array-like or sparse matrix, shape=(n_samples, n_features)
            Data points to assign to clusters based on distance metric
        '''
        centroids = [X[self.labels_==j].mean(axis=0) for j in range(self.n_clusters)]
        return np.array(centroids)

    def fit(self, X):
        ''''
        Compute k-means clustering.
        Parameters
        ----------
        X : array-like or sparse matrix, shape=(n_samples, n_features)
            Training instances to cluster.
        '''
        self._initialize_centroids(X)
        for i in range(self.max_iter):
            self._assign_clusters(X)
            new_centroids = self._compute_centroids(X)
            if ~(self.centroids_ != new_centroids).any():
                if self.verbose:
                    print('Converged on interation {}'.format(i))
                break
            # re-assign centroids
            self.centroids_ = new_centroids

    def predict(self, X):
        '''
        Optional method: predict the closest cluster each sample in X belongs to.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            New data to predict.
        Returns
        -------
        labels : array, shape [n_samples,]
            Index of the cluster each sample belongs to.
        '''
        labels = np.zeros(X.shape[0])
        for i, row in enumerate(X):
            labels[i] = ((row - self.centroids_)**2).sum(axis = 1).argmin()
        return labels.astype(int)

    def score(self, X):
        '''
        return the total residual sum of squares
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            New data.
        Returns
        -------
        score : float
            The SSE
        '''
        SSE = 0
        labels = self.predict(X)
        for label in np.unique(labels):
            SSE += ((X[labels == label] - self.centroids_[label])**2).sum()
        return SSE


def load_data():
    '''
    loads iris data set and returns iris data set as a np array
    '''
    iris = datasets.load_iris()
    return iris['data']


def elbow_plot(data, plotname):
    plt.clf()
    ks = list(range(2, 7))
    #np.arange(2, 7).astype(int)
    sses = []
    for k in ks:
        model = KMeans(n_clusters = k)
        model.fit(data)
        sses.append(model.score(data))
    plt.plot(ks, sses)
    plt.xlabel('Number of clusters')
    plt.ylabel('SSE')
    plt.title('Elbow Plot')
    plt.savefig(plotname)


def silhouette(data, k):
    model = KMeans(n_clusters = k)
    model.fit(data)
    labels = model.labels_
    return silhouette_score(data, labels, metric='euclidean')

if __name__ == '__main__':
    iris = load_data()

    elbow_plot(iris, 'elbow_plot.png')
    # has an issue with more than 7 centroids
    for k in range(2, 7):
        print(silhouette(iris, k))
