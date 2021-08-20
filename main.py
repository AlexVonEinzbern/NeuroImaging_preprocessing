import numpy as np
from sklearn.cluster import KMeans
from scipy import ndimage
import ants
#import antspynet

def denoise_image(image, k):
    """
    Parameters 
        image: ndarray
        k: size of neighbour
    
    Description: Apply median filter. 

    Returns: ndarray
    """
    return ndimage.filters.median_filter(image, size=k)
    
def normalization_zscore(image):
    """
    Parameters: 3D numpy array

    Description: Apply z-score normalization

    Returns: 3D numpy array
    """
    for i in range(image.shape[0]):
        zscore = (image-image[image>0].mean())/image[image>0].std()
    return zscore

def kmeans_segmentation(data, k):
    """
    Parameters: 3D numpy array
                k: number of clusters

    Description: Apply k-means algorithm to data

    Returns: 3D numpy
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

"""
def brain_extraction(image, modality):
    img = ants.from_numpy(image)
    probabilitu_brain_mask = antspynet.utilities.brain_extraction(img, modality=modality)
    return probabilitu_brain_mask
"""