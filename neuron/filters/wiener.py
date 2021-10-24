from scipy.signal.signaltools import wiener

def wiener(img, k):
    """
    Parameters: image: ndarray
                k: size of neighbours

    Description: Apply Wiener filter.

    Returns: ndarray
            wiener filtered result with the same shape as image
    """
    return wiener(img, k)