# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 09:42:43 2017

@author: h501zgrl
"""
import matplotlib.pyplot as plt
import numpy as np
from DicomRead import DicomRead
from compute_suv import compute_suv
from ViewTools import ClickAndRoi
from SegmentTools import createROI, segmentImage, createROI_from_mask
from TextureCalculs import histogram_TI, GLCM_TI, GLRLM_TI, GLZSM_TI
import os
from readimg import readimg_file
#import gui
   # return (currentImage)
import SimpleITK as sitk


def main(dicomPath, masked_Volume = None, verbose = 0):

  
    sData = DicomRead(dicomPath)
    # Im = sData[0][2]

    SUV = compute_suv(sData[1][0], sData[0][2])
#    plt.imshow(SUV[:, :, 202], cmap ='binary')   
#    fig = plt.gcf()
#    ax = plt.gca() 
#    myROI = ClickAndRoi(fig, ax)
#    print(myROI.selected_pixel)
    #plt.close()
    if masked_Volume is None:
        segmented_Im, mask_Image, _ = createROI(SUV, 202, [135, 133])
    else:
        segmented_Im, mask_Image = segmentImage(SUV, masked_Volume)
    #segmented_Im=segmented_Im*mask_Image
#    for i in range(1,10):
#        plt.figure()
#        plt.imshow(segmented_Im[:, :, i], cmap ='gist_gray') 

    # Some pre-processing steps
    num_img_values = 64
#    print(np.asarray(masked_Volume).shape)
#    plt.figure(dpi = 300)
    suvmax = np.max(segmented_Im[np.where(mask_Image == 1)])  
    suvmin = np.min(segmented_Im[np.where(mask_Image == 1)])
    suvmean = np.mean(segmented_Im[np.where(mask_Image == 1)])
    suvstd = np.std(segmented_Im[np.where(mask_Image == 1)])
    num_ROI_voxels = np.shape(np.argwhere(mask_Image == 1))[0]   
    
    
    # Volume
    pixelW = sData[0][0]['PixelWidth']
    pixelS = sData[0][0]['SliceThickness']   
    ROI_volume = num_ROI_voxels*(pixelW**2)*pixelS/1000
    


        
    patientName = str(sData[1][0].PatientName)
    seriesName = str(sData[1][0].SeriesDescription)
    print('Series = ', seriesName)
 
    if verbose is 1:
        print("\n***General information***")
        print('Patient = ', patientName)
        print('Series = ', seriesName)
    
    varnames = ['Patient', 'Series']
    to_save = [patientName]+[seriesName]
    # For the project
    try:
        beta_value = sData[1][0]['0009', '10f8'].value
    except:
        beta_value = 0
    print('SUVmax = ',"%.2f" %  suvmax)
    print('N. voxels = ',"%.2f" %  num_ROI_voxels)
    
    if verbose is 1:
        print("\n***ROI values***")
        print("Beta value =", beta_value)        
        print('SUVmax = ',"%.2f" %  suvmax)
        print('SUVmin = ',"%.2f" %  suvmin)
        print('SUVmean = ',"%.2f" %  suvmean)
        print('SUVstd = ', "%.2f" % suvstd)
        print('N. voxels = ',"%.2f" %  num_ROI_voxels)
        print('Volume = ', "%.2f" % ROI_volume, ' mL')
    
    NoneTI = np.array([0,
                       suvmax,
                       suvmin,
                       suvmean,
                       suvstd,
                       num_ROI_voxels,
                       ROI_volume])

    to_save += NoneTI.tolist()   
    varnames += ['beta', 'SUVmax', 'SUVmin', 'SUVmean', 'SUVstd', 'voxels', 'Volume']
    
    #Discretize the image 
    ## Im_min = suvmin
    ## Im_max = suvmax
                       
    # Absolute resampling
    #seg_Im_resc = num_img_values*(segmented_Im - Im_min) / (Im_max - Im_min)
    
    # Relative resampling (See Orlhac et al Plos One 2015) from 0 to 20
    seg_Im_resc = num_img_values*segmented_Im/20
    
    #add 1 to obtain the values from 1,...,N+1      
    seg_Im_resc = np.floor(seg_Im_resc) + 1

    # The maximum value is one unit higher than the maximum, so we put those
    #voxels to the maximum   
    seg_Im_resc[seg_Im_resc == num_img_values + 1] = num_img_values
 
    TI_hist, hist_varnames = histogram_TI(seg_Im_resc,
                                          mask_Image,
                                          num_img_values,
                                          num_ROI_voxels,
                                          verbose) 
    to_save += TI_hist.tolist()    
    varnames += hist_varnames

    TI_GCLM, varnames_GLCM = GLCM_TI(seg_Im_resc,
                                     mask_Image,
                                     num_img_values,
                                     verbose)  
    to_save += TI_GCLM.tolist()       
    varnames += varnames_GLCM

    TI_GLRM, varnames_GLRM = GLRLM_TI(seg_Im_resc,
                                      mask_Image,
                                      num_img_values,
                                      verbose)
    to_save += TI_GLRM.tolist()   
    varnames += varnames_GLRM      
    
    TI_GLZSM, varnames_GLZSM = GLZSM_TI(seg_Im_resc,
                                        mask_Image,
                                        num_img_values,
                                        num_ROI_voxels)
    to_save += TI_GLZSM.tolist()    
    varnames += varnames_GLZSM      

    
   
    return to_save, varnames


#to_save = [np.arange(1,15)]


#treshold_Image = readimg(path0)

# Find mask
# List of directories

import pandas as pd

wb_list = ['WB19']#['WB05''WB06','WB07','WB08', 'WB10','WB11','WB12','WB13','WB14', 'WB15', 'WB17', 'WB18']
#problems wb09, 'WB05',
#wb_list = ['phantom']        
#wb_list = [#'8_1_180_450',
#           #'8_1_180_400',
#          # '8_1_180_350',
#           #'8_1_180_300',
#           '8_1_120_450',
#           '8_1_120_400',
#           '8_1_120_350',
#           '8_1_120_300',
#           '8_1_60_450',
#           '8_1_60_400',
#           '8_1_60_350',
#           '8_1_60_300',
#           '4_1_120_450',
#           '4_1_120_400',
#           '4_1_120_350',
#           '4_1_120_300',
#           '2_1_120_450',
#           '2_1_120_400',
#           '2_1_120_350',
#           '2_1_120_300']

for iwb in wb_list:
#    dPath = 'C:/Users/h501zgrl/Documents/Projectes/heterogeneity/dataO/' + iwb
#    dPathMask = 'C:/Users/h501zgrl/Documents/Projectes/heterogeneity/data/' + iwb
    dPath = 'C:/Users/h501zgrl/Documents/Projectes/EANM_17_estudi_pacients/Dades2/' + iwb
    dPathMask = 'C:/Users/h501zgrl/Documents/Projectes/EANM_17_estudi_pacients/Dades/' + iwb
#dicomPath = 'C:/Users/h501zgrl/Documents/Projectes/EANM_17_estudi_pacients/Dades2/WB09/SER00000'   

#f = open(sData[1][0].PatientName.original_string  +".txt", "ab")
    ROIvalue = 1
    print('\n ', dPath)
    for filename in os.listdir(dPath):        
    #    if filename.endswith(".img"):
        if filename.endswith(".gz"):
   

           # treshold_Image = treshold_Image[::-1,::1,:]        

           # treshold_Image = np.fliplr(treshold_Image)
            # treshold_Image =  np.transpose(treshold_Image,(0,1,2))
            print('***ROI nº:', ROIvalue)

            for idicom in range(0,10):
                treshold_Image = readimg_file(os.path.join(dPath, filename))
                treshold_Image = np.rot90(treshold_Image, 1)
                treshold_Image = treshold_Image[:,::-1,:]  
                dicomPath = dPath + '/SER000'+ str(idicom).zfill(2)
                print(str(idicom).zfill(2)) 
                #' Compute mask from ROI
                sData = DicomRead(dicomPath)
                if str(sData[1][0].SeriesDescription) != 'q50':
                    SUV = compute_suv(sData[1][0], sData[0][2])                
                    _, _, treshold_Image2 = createROI_from_mask(SUV, treshold_Image)
                    
    #                import copy as copy                
    #                prova = copy.deepcopy(SUV)
    #                prova[np.where(treshold_Image == 0)] = 0
    #                plt.figure()
    #                plt.imshow(prova[:, :, 124], cmap ='gist_gray')
    #                plt.figure()
    #                plt.imshow(SUV[:, :, 124], cmap ='gist_gray')
    
    
                    to_save, varnames = main(dicomPath, treshold_Image2)                 
                    to_save = [ROIvalue] + to_save
                    varnames = ['ROI'] + varnames
                    varnames = varnames
                    to_save = [to_save]
                    my_df = pd.DataFrame(to_save, columns = varnames)
                    fileToSave = '../out/prova7.csv'
                    my_df.to_csv(fileToSave,
                                 header=(not os.path.exists(fileToSave)),
                                 mode='a')            
            ROIvalue = ROIvalue + 1
