from ants import from_numpy
from ants import atropos
from ants import get_mask

def atropos(img, k, m, c, priorweight):
    """
    
    """
    data = from_numpy(img)
    mask = get_mask(data)
    seg = atropos(a=data, m=m, c=c, i=f'kmeans[{k}]', x=mask)
    return atropos(a=data, m=m, c=c, i=seg['probabilityimages'], x=mask, priorweight=priorweight)
