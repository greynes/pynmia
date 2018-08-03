# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 13:06:22 2017

@author: h501zgrl
"""

def readimg_file(path):
    import nibabel as nib
    import numpy as np
    
    img = nib.load(path)#seg = sitk.ConnectedThreshold(sitk.GetImageFromArray(SUV), seedList=[(10, 10, 10)], lower=0, upper=1000)
#    a = np.array(img.dataobj)
   
#    with nib.openers.Opener(path) as fobj:
#        hdr = nib.Nifti1Header.from_fileobj(fobj, check=False)  # don't raise error
#        hdr.get 
#        hdr.set_dim_info(slice = 1)
#        data = hdr.data_from_fileobj(fobj)
#    hdr['magic'] = b'n+1\x00'
#    img = nib.Nifti1Image(data, None, hdr)
    a = np.array(img.dataobj)   
   # anat_img_data = img.get_data()
#    print(a.shape)
    a = a.astype('float64')
    a[a>0] = 1
    mask = a[:,:,:,0]
 
    return mask
    
    
