import numpy as np
from sklearn.cluster import KMeans
from scipy import ndimage
import ants
import antspynet

def median_filter(image, k):
    """
    Parameters: image: ndarray
                k: size of neighbour
    
    Description: Apply median filter. 

    Returns: ndarray
    """
    return ndimage.filters.median_filter(image, size=k)
    
def normalization_zscore(image):
    """
    Parameters: image : ndarray

    Description: Apply z-score normalization

    Returns: ndarray
    """
    for i in range(image.shape[0]):
        zscore = (image-image[image>0].mean())/image[image>0].std()
    return zscore

def kmeans_segmentation(data, k):
    """
    Parameters: data : ndarray
                k : number of clusters

    Description: Apply k-means algorithm to data

    Returns: ndarray
    """
    x, y, _ = plt.hist(data.ravel())
    tissue = data[data>np.min(x)]
    segTissue = KMeans(n_clusters=k,
            init='k-means++',
            n_init=10,
            max_iter=300,
            random_state=0)
    segTissue.fit(tissue.reshape(-1,1))
    seg = np.zeros(data.shape)
    seg[data>0]=segTissue.labels_+1
    return seg

def get_brain_mask(image, modality, th):
    """
    Parameters: image : ndarray
                modality:
                    "t1"
                    "t1v0"
                    "t1nobrainer"
                    "t1combined"
                    "flair"
                    "t2"
                    "bold"
                    "fa"
                    "t1t2infant"
                    "t1infant"
                    "t2infant"
                th : threshold

    Description: Probability brain image 
    Returns: ndarray 
    """
    img = ants.from_numpy(image)
    mask = antspynet.utilities.brain_extraction(img, modality=modality)
    image[mask<th]
    return image.numpy()