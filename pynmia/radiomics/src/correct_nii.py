# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 12:12:45 2017

@author: Marga
"""
import nibabel as nib
import numpy as np
import os
# The following just selects the right object to open the file - in
# this case zipfile.ZipFile
wb_list = ['WB05','WB06','WB07','WB08', 'WB10','WB11','WB12','WB13','WB14', 'WB15', 'WB17', 'WB18', 'WB19']

for iwb in wb_list:
#    dPath = 'C:/Users/h501zgrl/Documents/Projectes/heterogeneity/dataO/' + iwb
#    dPathMask = 'C:/Users/h501zgrl/Documents/Projectes/heterogeneity/data/' + iwb
    dPath = 'C:/Users/h501zgrl/Documents/Projectes/EANM_17_estudi_pacients/Dades2/' + iwb
    dPathMask = 'C:/Users/h501zgrl/Documents/Projectes/EANM_17_estudi_pacients/Dades/' + iwb
#dicomPath = 'C:/Users/h501zgrl/Documents/Projectes/EANM_17_estudi_pacients/Dades2/WB09/SER00000'   

#f = open(sData[1][0].PatientName.original_string  +".txt", "ab")
    ROIvalue = 1
    print('\n ', dPath)
    for filename in os.listdir(dPathMask):        

        if filename.endswith(".gz"):            
            print('***ROI nº:', ROIvalue)

            fname = os.path.join(dPathMask, filename)
            with nib.openers.Opener(fname) as fobj:
                hdr = nib.Nifti1Header.from_fileobj(fobj, check=False)  # don't raise error
                hdr.get 
                hdr.set_dim_info(slice = 1)
                hdr.set_sform(np.diag([3, 4, 5, 1]), code='scanner')
   
                hdr.set_qform(np.diag([3, 4, 5, 1]),'scanner')
                print(hdr.get_qform(coded=True))

                data = hdr.data_from_fileobj(fobj)
            hdr['magic'] = b'n+1\x00'
            img = nib.Nifti1Image(data, None, hdr)
            a = np.array(img.dataobj)
   # anat_img_data = img.get_data()
            print(a.shape)

            nib.save(img, os.path.join(dPath, filename))      
          #  treshold_Image = np.fliplr(treshold_Image)
          #  treshold_Image =  np.transpose(treshold_Image,(0,1,2)).shape
            ROIvalue = ROIvalue + 1
            