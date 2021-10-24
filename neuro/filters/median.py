from scipy.ndimage.filters import median_filter

def median(img, k):
    """
    Parameters: img: ndarray
                k: size of neighbour
    
    Description: Apply median filter. 

    Returns: ndarray
    """
    return median_filter(img, size=k)