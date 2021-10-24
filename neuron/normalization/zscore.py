def normalization_zscore(img):
    """
    Parameters: img : ndarray

    Description: Apply z-score normalization

    Returns: ndarray
    """
    for i in range(img.shape[0]):
        zscore = (img[img>0]-img[img>0].mean())/img[img>0].std()
    return zscore