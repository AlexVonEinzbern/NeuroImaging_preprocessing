from ants import from_numpy
from antspynet.utilities import brain_extraction

def get_brain_mask(img, modality, th):
    """
    NEED NVIDIA GPU
    Parameters: img : ndarray
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
    data = from_numpy(img)
    mask = brain_extraction(data, modality=modality)
    img[mask<th]
    return img.numpy()