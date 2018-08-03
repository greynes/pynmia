# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 14:16:37 2017

@author: h501zgrl
"""
import numpy as np


def histogram_TI(seg_Im_resc, mask_Image, num_img_values, num_ROI_voxels, verbose = 0):
        
    # Histogram metrics
    vox_val_hist = np.zeros(num_img_values)
    for voxel_value in range(1,num_img_values):
        pixels = np.shape(np.argwhere(np.logical_and(seg_Im_resc == voxel_value, mask_Image == 1)))[0]
        vox_val_hist[voxel_value] = pixels
   # print('pixels =', pixels)

    # Histogram probabilities
    vox_val_probs = vox_val_hist / num_ROI_voxels
   # print('\n vox_val_hist =', vox_val_hist)

   # print(vox_val_hist)
   # print(vox_val_probs)

    # The numerical values of each histogram bin:  
    vox_val_indices = np.arange(1,num_img_values+1)
    vox_val_indices = vox_val_indices*0.3125
   # print('\n vox_val_indices =', vox_val_indices)
   #print('\n vox_val_probs =', vox_val_probs)

    # Mean
    hmean = np.sum(vox_val_indices*vox_val_probs)
    
    # Variance
    hvariance = np.sum( ((vox_val_indices - hmean)**2) * vox_val_probs )
    
    
    if hvariance > 0:
        # Skewness
        hskewness = np.sum(((vox_val_indices - hmean)**3) * vox_val_probs ) / (hvariance**(3/2))  
    
    # Kurtosis
        hkurtosis = np.sum( ((vox_val_indices - hmean)**4) * vox_val_probs ) / (hvariance**2)    
    else:
        hskewness = 0
        hkurtosis = 0

    #  Energy
    henergy = np.sum(vox_val_probs**2)
    # Entropy (NOTE: 0*log(0) = 0 for entropy calculations)
    hist_nz_bin_indices = np.where(vox_val_probs > 0) 
    hentropy = -1*np.sum(vox_val_probs[hist_nz_bin_indices] * np.log10(vox_val_probs[hist_nz_bin_indices]) )
    
    if verbose is 1:
        print("\n***Histogram Indices***")
        print("mean = ", hmean)
        print("variance = ","%.4f" %  hvariance)
        print("skewness = ", "%.4f" %hskewness)    
        print("kurtosis = ", "%.4f" %hkurtosis)
        print("H. Energy = ","%.4f" % henergy)
        print("H. Entropy = ","%.4f" % hentropy)
    
    TIhistogram = np.array([hmean,
                            hvariance,
                            hskewness,
                            hkurtosis,
                            henergy,
                            hentropy])
    return TIhistogram
    
    
def GLCM_TI_old(seg_Im_resc, mask_Image, num_img_values, num_ROI_voxels):
    """
               Angle(phi,theta)    Offset
                  ----------------    ------
                   (0,90)              (1,0,0)
                   (90,90)             (0,1,0)
                   (-,90)              (0,0,1)
                   (45,90)             (1,1,0)
                   (135,90)            (-1,1,0)
                   (90,45)             (0,1,1)
                   (90,135)            (0,1,-1)
                   (0,45)              (1,0,1)
                   (0,135)             (1,0,-1)
                   (45,54.7)           (1,1,1)
                   (135,54.7)          (-1,1,1)
                   (45,125.3)          (1,1,-1)
                   (135,125.3)         (-1,1,-1)  
    """
    GLCM = np.zeros((num_img_values, num_img_values, 13), dtype = 'float64')     
    gray_levels = np.arange(1, num_img_values + 1)
    

    
    offset = np.array([[1,0,0],
                       [0,1,0],
                       [0,0,1],
                       [1,1,0],
                       [-1,1,0],
                       [0,1,1],
                       [0,1,-1],
                       [1,0,1], 
                       [1,0,-1],
                       [1,1,1],
                       [-1,1,1],
                       [1,1,-1],
                       [-1,1,-1]])
    
        
#Get the indices for matches against 2. Use np.argwhere here to get those in a nice 2D array with each column representing an axis. Another benefit is that this makes it generic to handle arrays with generic number of dimensions. Then, add offset in a broadcasted manner. This is idx.
#Among the indices in idx, there would be few invalid ones that go beyond the array shape. So, get a valid mask valid_mask and hence valid indices valid_idx among them.
#Finally index into input array with those, compare against 3 and count the number of matches.    
    
#    
    for i, ivoxel_value in enumerate(gray_levels):
        i_indices = np.argwhere(np.logical_and(seg_Im_resc == ivoxel_value,  mask_Image == 1))
                               
        for j, jvoxel_value in enumerate(gray_levels):      
#                                   
            for ioffset, shift in enumerate(offset):
                idx = i_indices + shift
                valid_mask = (idx < seg_Im_resc.shape).all(1)
                valid_idx = idx[valid_mask]
                count = np.count_nonzero(
                        np.logical_and(
                           seg_Im_resc[tuple(valid_idx.T)] == jvoxel_value,
                           mask_Image[tuple(valid_idx.T)] == 1))
                 
                GLCM[i, j, ioffset] = count
                
    # Add the transpose to obtain a symetric matrix 
    for ioffset, shift in enumerate(offset): 
        GLCM[:,:,ioffset] += GLCM[:,:,ioffset].transpose()
        
    # Normalization
    nGLCM = np.mean(GLCM, axis = 2)
 

    nGLCM = nGLCM/np.sum(nGLCM)
  
    # Create meshgrid to compute
    i, j = np.meshgrid(gray_levels, gray_levels,  indexing='ij')
  #  print(num_img_values)
    homogeneity = np.sum(nGLCM / (1 + (np.abs(i - j))))

    # Energy / Angular second moment
    energy = np.sum( nGLCM**2 );

    #Compute p_{x+y}   
    p_xpy = np.zeros((2*num_img_values,1))
    for this_row in np.arange(0,num_img_values):
        for this_col in np.arange(0,num_img_values):
            p_xpy[this_row+this_col] = p_xpy[this_row+this_col] + nGLCM[this_row,this_col];
        
    entropy = -np.sum(p_xpy[p_xpy>0] * np.log10(p_xpy[p_xpy>0]))
   # entropy = -np.sum(nGLCM[nGLCM>0]*np.log10(nGLCM[nGLCM>0]))
    
    contrast = np.sum(nGLCM *((i - j)**2))
    dissimilarity = np.sum(nGLCM *(np.abs(i - j)))

    #Compute marginal distibutions
    p_x = np.sum(nGLCM,1, keepdims = True)
    p_y = np.sum(nGLCM,0, keepdims = True)
#    
    mu_x = np.sum(i*p_x)
    mu_y = np.sum(j*p_y)
# TODO: if sg_x or sg_y = 0
    sg_x = np.sqrt(np.sum(( ((i - mu_x)**2 ) * p_x )))
    sg_y = np.sqrt(np.sum(( ((j - mu_y)**2 ) * p_y )))
    
    #correlation = np.sum(nGLCM *((j[:,:] - mu_y)*(i-mu_x))/(sg_x*sg_y))   

    correlation = (np.sum((i-mu_x)*(j-mu_y)*nGLCM)/(sg_x*sg_y))
    print("\n***GLCM Indices***")   
    print('Homogeneity: ',"%.4f" % homogeneity)   
    print('Energy: ',"%.4f" % energy)   
    print('Contrast: ',"%.4f" % contrast)
    print('Correlation: ',"%.4f" % correlation)
    print('Sum. Entropy: ',"%.4f" % entropy)
    print('Dissimilarity: ', "%.4f" % dissimilarity)
    
    TIGLCM = np.array([homogeneity,
                       energy,
                       contrast,
                       correlation,
                       entropy,
                       dissimilarity])
    return TIGLCM
    
    
    
def GLRLM_TI(seg_Im_resc, mask_Image, num_img_values, verbose = 0): 
     """
     Gray Level Run Length Matrix (GLRLM) gives the size of the gray level runs
     """
     
     
     maxSizeRuns = np.max(seg_Im_resc.shape)
     GLRLM = np.zeros((num_img_values, maxSizeRuns, 13), dtype = 'float64')     
     gray_levels = np.arange(1, num_img_values + 1)
    

    
     offset = np.array([[1,0,0],
                        [0,1,0],
                        [0,0,1],
                        [1,1,0],
                        [-1,1,0],
                        [0,1,1],
                        [0,1,-1],
                        [1,0,1], 
                        [1,0,-1],
                        [1,1,1],
                        [-1,1,1],
                        [1,1,-1],
                        [-1,1,-1]])
    
        
#Get the indices for matches against 2. Use np.argwhere here to get those in a nice 2D array with each column representing an axis. Another benefit is that this makes it generic to handle arrays with generic number of dimensions. Then, add offset in a broadcasted manner. This is idx.
#Among the indices in idx, there would be few invalid ones that go beyond the array shape. So, get a valid mask valid_mask and hence valid indices valid_idx among them.
#Finally index into input array with those, compare against 3 and count the number of matches.    
    
#    max
     
     for i, ivoxel_value in enumerate(gray_levels):
         indices = np.argwhere(np.logical_and(seg_Im_resc == ivoxel_value,
                                             mask_Image == 1))
     
         results = np.zeros((maxSizeRuns, 13))            
         
         for idire, dire in enumerate(offset):
            results[0, idire] = len(indices)
            indices_to_compare = indices 
             
            for irun in range(1, maxSizeRuns):
                indices_displaced = indices + dire*irun             
                aset = set([tuple(x) for x in indices_displaced])
                bset = set([tuple(x) for x in indices_to_compare])
                indices_to_compare = (np.array([x for x in aset & bset]))
                results[irun, idire] = len(indices_to_compare)
                            
            for iindx in np.arange(maxSizeRuns-2,-1,-1):
                for jindx in np.arange(iindx+1, maxSizeRuns):
                    results[iindx, idire] -=  results[jindx, idire] *(jindx-iindx+1)
                 
         GLRLM[i, :, :] = results
                 
     
    # Computation of indices


    # Coefficients
     pr = np.sum(GLRLM, 0)
     pg = np.sum(GLRLM, 1)
     jvector = np.arange(1, GLRLM.shape[1] + 1, dtype=np.float64)
     
     emptyGrayLevels = np.where(np.sum(pg, 1) == 0)
     emptyRunLenghts = np.where(np.sum(pr, 1) == 0) 
     
     GLRLM = np.delete(GLRLM, emptyGrayLevels, 0)
     GLRLM = np.delete(GLRLM, emptyRunLenghts, 1)
     
     
     jvector = np.delete(jvector, emptyRunLenghts)
     ivector = np.delete(gray_levels, emptyGrayLevels)
     pg = np.delete(pg, emptyGrayLevels, 0)
     pr = np.delete(pr, emptyRunLenghts, 0)


     sumP_GLRLM = np.sum(GLRLM, (0, 1))
     emptyAngles = np.where(sumP_GLRLM == 0)

     GLRLM = np.delete(GLRLM, emptyAngles, 2)
     sumP_GLRLM = np.delete(sumP_GLRLM, emptyAngles, 0)
     
     
     
     # Short Run Emphasis     
     sre = np.sum((pr/(jvector[:, None]**2)), 0)/sumP_GLRLM
     sre = sre.mean() 
     
     # Long Run Emphasis: 
     lre = np.sum((pr * (jvector[:, None] ** 2)), 0) / sumP_GLRLM
     lre = lre.mean()
     
     
     # Low  Gray-level Run Emphasis: 
     lglre = np.sum((pg / (ivector[:, None] ** 2)), 0) / sumP_GLRLM
     lglre = lglre.mean()
     
     # High  Gray-level Run Emphasis: 
     hglre = np.sum((pg * (ivector[:, None] ** 2)), 0) / sumP_GLRLM
     hglre = hglre.mean()
     
     if verbose is 1:
         print("\n***GLRM Indices***") 
         print('SRE: ',"%.4f" % sre)   
         print('LRE: ',"%.4f" % lre)
         print('LGRE', "%.4f" % lglre)
         print('HGRE', "%.4f" % hglre)
     
     TI_GLRLM = np.array([sre,
                          lre,
                          lglre,
                          hglre])
    
     return TI_GLRLM
    
    
def GLZSM_TI(seg_Im_resc, mask_Image, num_img_values, verbose = 0):
    """
     The gray-level zone length matrix (GLZLM) [Thibault] provides information on the
    size of homogenous zones for each gray-level in 3 dimensions.
    """              
    from skimage import measure
    GLZSM = np.zeros((num_img_values, seg_Im_resc.size), dtype = 'float64')     
    gray_levels = np.arange(1, num_img_values + 1)
    
    
    for i, ivoxel_value in enumerate(gray_levels):
         graylevelMask = np.zeros(np.shape(seg_Im_resc), dtype = np.int)
         indices = np.where(np.logical_and(seg_Im_resc == ivoxel_value,
                                             mask_Image == 1))
        # print('\n', ivoxel_value, indices)                                    
         graylevelMask[indices] = 1
         np.count_nonzero(graylevelMask)
         labeledMask, numlabels = measure.label(graylevelMask,
                                                background  = 0,
                                                return_num = True)
         for ilabel in range(0, numlabels):
             labelpixels = np.count_nonzero(labeledMask == ilabel)
             GLZSM[i, labelpixels] += 1
             
    # Coefficients
    pr = np.sum(GLZSM, 0)
    pg = np.sum(GLZSM, 1)
    jvector = np.arange(1, GLZSM.shape[1] + 1, dtype = np.float64)
     
    emptyGrayLevels = np.where(pg == 0)
    emptySizeLenghts = np.where(pr == 0) 
     
    GLZSM = np.delete(GLZSM, emptyGrayLevels, 0)
    GLZSM = np.delete(GLZSM, emptySizeLenghts, 1)
    jvector = np.delete(jvector, emptySizeLenghts)
    ivector = np.delete(gray_levels, emptyGrayLevels)
    pg = np.delete(pg, emptyGrayLevels)
    pr = np.delete(pr, emptySizeLenghts)
    
    sumP_GLZSM = np.sum(GLZSM, (0, 1))
    # Small area emphasis
    sae = np.sum(pr / (jvector ** 2)) / sumP_GLZSM
    lae = np.sum(pr * (jvector ** 2)) / sumP_GLZSM
    
    # Low Gray-level Zone Emphasis
    lgze = np.sum((pg / (ivector ** 2))) / sumP_GLZSM
    hgze = np.sum((pg * (ivector ** 2))) / sumP_GLZSM

    if verbose is 1:
        print("\n***GLZSM Indices***") 
        print('SAE: ',"%.4f" % sae)   
        print('LAE: ',"%.4f" % lae)
        print('LGZE', "%.4f" % lgze)
        print('HGZE', "%.4f" % hgze)
    
    varnames = ['SRE', 'LRE', 'LGRE', 'HGRE']
  
    TIGLZSM = np.array([sae,
                        lae,
                        lgze,
                        hgze])
                        
    return TIGLZSM, varnames
        
        
def GLCM_TI(seg_Im_resc, mask_Image, num_img_values, verbose = 0):
    """
               Angle(phi,theta)    Offset
                  ----------------    ------
                   (0,90)              (1,0,0)
                   (90,90)             (0,1,0)
                   (-,90)              (0,0,1)
                   (45,90)             (1,1,0)
                   (135,90)            (-1,1,0)
                   (90,45)             (0,1,1)
                   (90,135)            (0,1,-1)
                   (0,45)              (1,0,1)
                   (0,135)             (1,0,-1)
                   (45,54.7)           (1,1,1)
                   (135,54.7)          (-1,1,1)
                   (45,125.3)          (1,1,-1)
                   (135,125.3)         (-1,1,-1)  
    """
    GLCM = np.zeros((num_img_values, num_img_values, 13), dtype = 'float64')     
    gray_levels = np.arange(1, num_img_values + 1)
        
    offset = np.array([[1,0,0],
                       [0,1,0],
                       [0,0,1],
                       [1,1,0],
                       [-1,1,0],
                       [0,1,1],
                       [0,1,-1],
                       [1,0,1], 
                       [1,0,-1],
                       [1,1,1],
                       [-1,1,1],
                       [1,1,-1],
                       [-1,1,-1]])
    
        
#Get the indices for matches against 2. Use np.argwhere here to get those in a nice 2D array with each column representing an axis. Another benefit is that this makes it generic to handle arrays with generic number of dimensions. Then, add offset in a broadcasted manner. This is idx.
#Among the indices in idx, there would be few invalid ones that go beyond the array shape. So, get a valid mask valid_mask and hence valid indices valid_idx among them.
#Finally index into input array with those, compare against 3 and count the number of matches.    
    
#    
    for i, ivoxel_value in enumerate(gray_levels):
        i_indices = np.argwhere(np.logical_and(seg_Im_resc == ivoxel_value,  mask_Image == 1))
                               
        for j, jvoxel_value in enumerate(gray_levels):      
#                                   
            for ioffset, shift in enumerate(offset):
                idx = i_indices + shift
                valid_mask = (idx < seg_Im_resc.shape).all(1)
                valid_idx = idx[valid_mask]
                count = np.count_nonzero(
                        np.logical_and(
                           seg_Im_resc[tuple(valid_idx.T)] == jvoxel_value,
                           mask_Image[tuple(valid_idx.T)] == 1))
                 
                GLCM[i, j, ioffset] = count
                
    # Add the transpose to obtain a symetric matrix 
    for ioffset, shift in enumerate(offset): 
        GLCM[:,:,ioffset] += GLCM[:,:,ioffset].transpose()
        
    # Normalization
 
    sumP_GLCM = np.sum(GLCM, (0, 1))
    nGLCM = GLCM/sumP_GLCM
  
    # Create meshgrid to compute
    i, j = np.meshgrid(gray_levels, gray_levels,  indexing='ij')
  #  print(num_img_values)
    homogeneity = np.sum((nGLCM / (1 + (np.abs(i - j))[:, :, None])), (0, 1))
    homogeneity = np.mean(homogeneity)
    
    energy = np.sum(nGLCM**2, (0,1))
    energy = np.mean(energy)
#    # Energy / Angular second moment
#    energy = np.sum( nGLCM**2 );
#
#    #Compute p_{x+y}   
#    p_xpy = np.zeros((2*num_img_values,1))
#    for this_row in np.arange(0,num_img_values):
#        for this_col in np.arange(0,num_img_values):
#            p_xpy[this_row+this_col] = p_xpy[this_row+this_col] + nGLCM[this_row,this_col];
#        
#    entropy = -np.sum(p_xpy[p_xpy>0] * np.log10(p_xpy[p_xpy>0]))
    entropy = -np.sum(nGLCM*np.log10(nGLCM+np.finfo(np.double).tiny ), (0, 1))
    entropy = np.mean(entropy)
    contrast = np.sum((nGLCM * ((i - j)[:, :, None] ** 2)), (0, 1))
    contrast = np.mean(contrast)
    dissimilarity = np.sum((nGLCM * ((np.abs(i - j))[:, :, None])), (0, 1))
    dissimilarity = np.mean(dissimilarity)   
    
    #Autocorrelation
    autocorrelation = np.sum((nGLCM * ((i * j)[:, :, None])), (0, 1)) 
    #dissimilarity = np.sum(nGLCM *(np.abs(i - j)))

#    #Compute marginal distibutions
    mu_x = np.sum(i[:, :, None] * nGLCM, (0, 1), keepdims=True)
    mu_y = np.sum(j[:, :, None] * nGLCM, (0, 1), keepdims=True)
##    
#   mu_x = np.sum(i*p_x, (0,1))
#  mu_y = np.sum(j*p_y, (0, 1))
## TODO: if sg_x or sg_y = 0
    #sg_x = np.sqrt(np.sum(( ((i - mu_x)**2 ) * p_x )))
    sg_x = (np.sum(nGLCM * ((i[:, :, None] - mu_x) ** 2), (0, 1), keepdims=True)) ** 0.5
    sg_y = (np.sum(nGLCM * ((j[:, :, None] - mu_x) ** 2), (0, 1), keepdims=True)) ** 0.5
#    
    correlation = np.sum(nGLCM *(j[:,:, None] - mu_y)*(i[:,:,None]-mu_x),(0,1),  keepdims=True)/(sg_x*sg_y) 
    correlation = np.mean(correlation)
    #correlation = (np.sum((i-mu_x)*(j-mu_y)*nGLCM)/(sg_x*sg_y))
    if verbose is 1:    
        print("\n***GLCM Indices***")   
        print('Homogeneity: ',"%.4f" % homogeneity)   
        print('Energy: ',"%.4f" % energy)   
        print('Contrast: ',"%.4f" % contrast)
        print('Correlation: ',"%.4f" % correlation)
        print('Sum. Entropy: ',"%.4f" % entropy)
        print('Dissimilarity: ', "%.4f" % dissimilarity)
        print('Autoccorrelation:', "%.4f" % autocorrelation)
    
    varnames = [' Homogeneity',
                'Energy',
                'Contrast',
                'Correlation',
                'Entropy',
                'Dissimilarity',
                'Autocorrelation']  
#    TIGLCM = np.array([homogeneity,
#                       energy,
#                       contrast,
#                       correlation,
#                       entropy,
#                       dissimilarity])
    TIGLCM = np.array([homogeneity,
                       energy,
                       contrast,
                       correlation,
                       entropy,
                       dissimilarity,
                       autocorrelation])
    return TIGLCM, varnames    
    
    
def NGLDM_TI(seg_Im_resc, mask_Image, num_img_values, num_ROI_voxels, verbose = 0):
    """
    The neighborhood gray-level different matrix (NGLDM) corresponds 
    to the difference of gray-level between one voxel and its 26 neighbours in 
    3 dimensions.
    TODO: start
    """
  #  
    gray_levels = np.arange(1, num_img_values + 1)
 #  NGLDM = np.zeros((ggray_levels, gray_levels), dtype = 'float64')     


    