# -*- coding: utf-8 -*-
"""
Obtain SUV in gm/mL
"""
def compute_suv(dicom_headers, PET):
    import numpy as np
    
    def hms2seconds(time_string):
        h = float(time_string[0:2])
        m = float(time_string[2:4])
        s = float(time_string[4:6])
        time = h*3600 + m*60 + s   
        return time
            
    PET = np.asarray(PET, dtype = 'float64')
    # Get patient weight in grams
    try:
        weight = float(dicom_headers.PatientWeight*1000) 
    except ValueError:
        print('None patient weight found assuming 70 kg')
        weight = 70.0 
        
    # Get the acquisition time
    try:
        scan_time = hms2seconds(dicom_headers.AcquisitionTime)
    except ValueError:
        print('Error') 
        
    # Radiopharmaceutical injection time
    injection_time = hms2seconds(dicom_headers.
                                 RadiopharmaceuticalInformationSequence[0].
                                 RadiopharmaceuticalStartTime)
    # Half Life
    half_life = float(dicom_headers.
                      RadiopharmaceuticalInformationSequence[0].
                      RadionuclideHalfLife)                  
    # Injected Dose
    injected_dose = float(dicom_headers.
                          RadiopharmaceuticalInformationSequence[0].
                          RadionuclideTotalDose)
    
    # Decayed Dose in Bq
    decayed_dose = injected_dose*np.exp(-np.log(2.0)*(scan_time-injection_time)/half_life)   
   
   # Compute SUV
    SUV = PET*weight/decayed_dose;
    
    
    return SUV