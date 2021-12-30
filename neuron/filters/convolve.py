from scipy import ndimage
from numpy import linspace
from scipy.stats import norm

def convolve(img, k):
	"""
	"""
	x = linspace(-k, k, (2*k)+1)
	kernel = norm.pdf(x)*norm.pdf(x[:,None])
	return ndimage.convolve(img, kernel[:,:,None])