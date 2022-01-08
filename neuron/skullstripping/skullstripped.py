from pyrobex.robex import robex

def skullstripped(img):
	"""
	Parameters: img : nibabel.NiftiImage
	
	Description: returns a pyrobex NifTI tuple
			* stripped : Image of the extracted brain
			* mask : Image of a binary mask of the brain  
  
	Returns: pyrobex.io.NiftiImage
	"""
	return robex(img)