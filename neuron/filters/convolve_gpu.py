import jax.scipy as jsp
from numpy import linspace

def convolve_gpu(img, k):
    """
    NEED GPU
    
    Parameters: img: ndarray
                k: size of neighbour
    
    Description: Apply convolve filter. 

    Returns: ndarray
    """
    x = linspace(-k, k, 2*k+1)
    kernel = jsp.stats.norm.pdf(x) * jsp.stats.norm.pdf(x[:, None])    
    return jsp.signal.convolve(img, kernel[:,:,None])