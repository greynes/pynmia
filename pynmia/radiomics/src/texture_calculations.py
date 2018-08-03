# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 14:16:37 2017

@author: h501zgrl
"""
import numpy as np

def histogram_TI(seg_Im_resc, mask_Image, num_img_values, num_ROI_voxels):
        
    # Histogram metrics
    vox_val_hist = np.zeros(num_img_values)
    for voxel_value in range(1,num_img_values):
        pixels = np.size(np.where(np.logical_and(seg_Im_resc == voxel_value, mask_Image == 1)))
        vox_val_hist[voxel_value] = pixels
       # print('pixels =', pixels)

    # Histogram probabilities
    vox_val_probs = vox_val_hist / num_ROI_voxels
    print('\n vox_val_hist =', vox_val_hist)

   # print(vox_val_hist)
   # print(vox_val_probs)

    # The numerical values of each histogram bin:  
    vox_val_indices = np.arange(1,num_img_values+1)
    vox_val_indices = vox_val_indices#*0.3125
    print('\n vox_val_indices =', vox_val_indices)
    print('\n vox_val_probs =', vox_val_probs)

    # Mean
    hmean = np.sum(vox_val_indices*vox_val_probs)
    print("mean = ",hmean)
    
    # Variance
    hvariance = np.sum( ((vox_val_indices - hmean)**2) * vox_val_probs )
    print("variance = ", hvariance)
    
    
    if hvariance > 0:
        # Skewness
        hskewness = np.sum(((vox_val_indices - hmean)**3) * vox_val_probs ) / (hvariance**(3/2))  
    
    # Kurtosis
        hkurtosis = np.sum( ((vox_val_indices - hmean)**4) * vox_val_probs ) / (hvariance**2)    
    else:
        hskewness = 0
        hkurtosis = 0
        
    print("skewness = ", hskewness)    
    print("kurtosis = ", hkurtosis)
    
    #  Energy
    henergy = np.sum(vox_val_probs**2)
    print("H. Energy = ",henergy)
    # Entropy (NOTE: 0*log(0) = 0 for entropy calculations)
    hist_nz_bin_indices = np.where(vox_val_probs > 0) 
    hentropy = -1*np.sum(vox_val_probs[hist_nz_bin_indices] * np.log10(vox_val_probs[hist_nz_bin_indices]) )
    print("H. Entropy = ", hentropy)