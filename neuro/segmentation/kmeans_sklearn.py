from sklearn.cluster import KMeans
from matplotlib.pyplot import hist
from numpy import min, zeros

def kmeans_sklearn(data, k):
    """
    Parameters: data : ndarray
                k : number of clusters

    Description: Apply k-means algorithm to data

    Returns: ndarray
    """
    x, y, _ = hist(data.ravel())
    tissue = data[data>min(x)]
    segTissue = KMeans(n_clusters=k,
            init='k-means++',
            n_init=10,
            max_iter=300,
            random_state=0)
    segTissue.fit(tissue.reshape(-1,1))
    seg = zeros(data.shape)
    seg[data>0]=segTissue.labels_+1
    return seg