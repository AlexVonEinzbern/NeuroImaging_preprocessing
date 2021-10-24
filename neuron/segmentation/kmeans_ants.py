from ants import from_numpy
from ants import kmeans_segmentation

def kmeans_ants(img, k):
    """
    Parameters: img : ndarray
                k : number of clusters

    Description: Apply k-means algorithm from antspyx library to img

    Returns: ANTsImage (See antspyx documentation)
    """
    data = from_numpy(img)
    return kmeans_segmentation(data, k)