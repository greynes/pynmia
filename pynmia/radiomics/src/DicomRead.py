# -*- coding: utf-8 -*-
"""
Read D
"""

def DicomRead(dicomPath):
    """
    This function read a Dicom dir and obtain the headers and volume.
    Tested just with GE Healthcare PET/CT DICOM
    TODO: read dicomRT
    """    
    import dicom
    import numpy as np
    import os 

    RTstruct = []
    REG = []
    # Read one file to get volume
    elements = os.listdir(dicomPath)
    nelements = len(elements)
    temp = dicom.read_file(os.path.join(dicomPath, elements[0]))
    if temp.Modality == 'PT':
            VolumeDimensions  = (int(temp.Rows), int(temp.Columns), nelements)
            ConstPixelSpacing = (float(temp.PixelSpacing[0]),
                                 float(temp.PixelSpacing[1]),
                                 float(temp.SliceThickness))
    
    x = np.arange(0.0,(VolumeDimensions[0]+1)*ConstPixelSpacing[0], ConstPixelSpacing[0])
    y = np.arange(0.0,(VolumeDimensions[1]+1)*ConstPixelSpacing[1], ConstPixelSpacing[1])
    z = np.arange(0.0,(VolumeDimensions[2]+1)*ConstPixelSpacing[2], ConstPixelSpacing[2])        
    volume = np.zeros(VolumeDimensions, dtype = 'float64')
    sliceNumber = 0
    dicomHeaders = []
    for filename in elements:
        elementPath = os.path.join(dicomPath, filename)
        temp = dicom.read_file(elementPath)
        volume[:,:, sliceNumber] = temp.pixel_array        
        temp.pixel_array = None
        dicomHeaders.append(temp)
        sliceNumber += 1
        
    # Obtain scan orientation
    dist = np.array([
            abs(dicomHeaders[1].ImagePositionPatient[0] -
                dicomHeaders[0].ImagePositionPatient[0]),
            abs(dicomHeaders[1].ImagePositionPatient[1] -
                dicomHeaders[0].ImagePositionPatient[1]),
            abs(dicomHeaders[1].ImagePositionPatient[2] -
                dicomHeaders[0].ImagePositionPatient[2])
            ])
    index = dist.argmax();
    if index == 0:
        orientation = 'Sagittal'
    elif index == 1:
        orientation = 'Coronal'
    elif index == 2:
        orientation = 'Axial'
    
    dicomHeaders = [dicomHeaders[i] for i in range(0, sliceNumber)]
    # Rescale Image (GE Healthcare)
    if temp.Modality == 'PT' or temp.Modality == 'CT':
        for i in range(1, sliceNumber):
            volume[:,:,i] = (volume[:,:,i]*dicomHeaders[i].RescaleSlope + 
                             dicomHeaders[i].RescaleIntercept)       
    
    # Sort images
    slicePosition = np.zeros(sliceNumber)
    for sliceIndex in range(0, sliceNumber):
        slicePosition[sliceIndex] = dicomHeaders[sliceIndex].ImagePositionPatient[index]
   # print(volume.shape)
    sortedIndices = slicePosition.argsort()
    #print(sortedIndices)
    volume = volume[:,:,sortedIndices]
    #print(volume.shape)

  


    # Fill sData
    sData = np.empty(2, dtype=object)  
    sData[0] = [{
                 'Modality': temp.Modality,
                 'Orientation': orientation,
                 'PixelWidth': ConstPixelSpacing[0],
                 'SliceThickness': ConstPixelSpacing[2],
                 'SliceSpacing': ConstPixelSpacing[1]
                 },
                 {'x':x,
                  'y':y,
                  'z':z},
                 volume]
    sData[1] = dicomHeaders    
    return sData
    
    
    
    
    
    
    
    