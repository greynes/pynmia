# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 11:07:17 2017

@author: h501zgrl

TODO: create a sphere insted a circle to find the maximum
TODO: accept RT strcuture
TODO: accept any treshold
TODO: option to close the image
"""
import numpy as np
import matplotlib.pyplot as plt
#from skimage import morphology
from skimage import measure
from copy import deepcopy
import SimpleITK as sitk

def createROI(currentImage, PETSlice, selected_pixel):
    # Create a ROI via treshold, return bounding box
    tresh = 0.41
   
    circle_masked_Im = deepcopy(currentImage[:, :, PETSlice])
    nx, ny = np.shape(circle_masked_Im)
    
    # Create a circle to find around the clicked position the maximum value
    cordx, cordy = np.ogrid[0:nx, 0:ny]
    circle_mask = ((cordx - selected_pixel[0])**2 +
                    (cordy - selected_pixel[1])**2 <  10)
    # Find the maximum
    SUVmax = np.amax(circle_masked_Im[circle_mask])
    np.argmax(circle_masked_Im[circle_mask])
    

    currentImageITK = sitk.GetImageFromArray(currentImage)
    seed = (PETSlice, selected_pixel[1], selected_pixel[0])
    
   
    seg = sitk.ConnectedThreshold(currentImageITK, seedList=[seed], lower=tresh*SUVmax, upper = 99, replaceValue = 1)
    seg2 = sitk.GetArrayFromImage(seg)
    xmin, xmax, ymin, ymax, zmin, zmax = boundingbox(seg2)

    mask_Image = seg2[xmin:xmax, ymin:ymax, zmin:zmax]
    
    segmented_Im = currentImage[xmin:xmax, ymin:ymax, zmin:zmax]    
    
    treshold_Image = seg2

    return segmented_Im, mask_Image, treshold_Image 
    
def createROI_from_mask(currentImage, masked_Volume, treshold = 0.41):
# Create a ROI via treshold, return bounding box
    tresh = treshold
   

    # Find the maximum
    SUVmax = np.amax(currentImage[np.where(masked_Volume == 1)])
    SUVmax_args = np.argwhere(np.logical_and(currentImage == SUVmax,
                                             masked_Volume == 1))
    SUVmax_args = SUVmax_args[0].tolist()    
#    print('a')
#    print(SUVmax)
    SUVmin = tresh*SUVmax
#    print(SUVmin)
#    currentImage_Seg = np.ones(currentImage.shape)
#
#    currentImage_Seg[np.where(currentImage < SUVmin)] = 0
    currentImageITK = sitk.GetImageFromArray(currentImage)
    seed = (SUVmax_args[2], SUVmax_args[1], SUVmax_args[0])
#    print(seed)
    #selected_pixel= [135, 133]
    #seed = (115, selected_pixel[1], selected_pixel[0])    
    #print(seed)
    
    seg = sitk.ConnectedThreshold(currentImageITK,
                                  seedList = [seed],
                                  upper = SUVmax,
                                  lower = SUVmin,
                                  connectivity = 1,
                                  replaceValue = 1)
    #sitk_show(seg)
                             
    seg2 = sitk.GetArrayFromImage(seg)
#    print('sum', seg2.sum())
#    print('max', seg2.max())

    xmin, xmax, ymin, ymax, zmin, zmax = boundingbox(seg2)

    mask_Image = seg2[xmin:xmax, ymin:ymax, zmin:zmax]
    
    segmented_Im = currentImage[xmin:xmax, ymin:ymax, zmin:zmax]    
    
    treshold_Image = seg2

    return segmented_Im, mask_Image, treshold_Image   

    return segmented_Im, mask_Image, treshold_Image 

    
def segmentImage(currentImage, masked_Volume):
    # Segment the image given a masked volume, return bounding box
    
    seg2 = masked_Volume

    xmin, xmax, ymin, ymax, zmin, zmax = boundingbox(seg2)
    mask_Image = seg2[xmin:xmax, ymin:ymax, zmin:zmax]

    segmented_Im = currentImage[xmin:xmax, ymin:ymax, zmin:zmax]    
 
    return segmented_Im, mask_Image 

    
def boundingbox(img):
    img = (img>0)
    # Compute the bounding box of the masked image
    x = np.any(img, axis=(1, 2))
    y = np.any(img, axis=(0, 2))
    z = np.any(img, axis=(0, 1))

    xmin, xmax = np.where(x)[0][[0, -1]]
    ymin, ymax = np.where(y)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]

    xmin = xmin 
    xmax = xmax +1
    ymin = ymin 
    ymax = ymax +1
    zmin = zmin 
    zmax = zmax +1
    return xmin, xmax, ymin, ymax, zmin, zmax
    
