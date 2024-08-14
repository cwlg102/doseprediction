import os
import SimpleITK as sitk 
import numpy as np
import pydicom
from multiprocessing import Process, Queue

def resample(sitk_volume, new_spacing, new_size, default_value=0, is_label=False):
    """1) Create resampler"""
    resample = sitk.ResampleImageFilter() 
    
    """2) Set parameters"""
    #set interpolation method, output direction, default pixel value
    resample.SetInterpolator(sitk.sitkLinear)
    resample.SetOutputDirection(sitk_volume.GetDirection())
    resample.SetDefaultPixelValue(default_value)
    
    #set output spacing
    new_spacing = np.array(new_spacing)
    resample.SetOutputSpacing(new_spacing)
    
    #set output size and origin
    old_size = np.array(sitk_volume.GetSize())
    old_spacing = np.array(sitk_volume.GetSpacing())
    new_size_no_shift = np.int16(np.round(old_size*old_spacing/new_spacing))
    old_origin = np.array(sitk_volume.GetOrigin())
    
    shift_amount = np.int16(np.round((new_size_no_shift - new_size)/2))*new_spacing
    new_origin = old_origin + shift_amount
    
    new_size = [int(s) for s in new_size]
    resample.SetSize(new_size)
    resample.SetOutputOrigin(new_origin)
    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        pass


    """3) execute"""
    new_volume = resample.Execute(sitk_volume)
    return new_volume


mr_basepath = r"G:\! project\2024- UnityDosePrediction\data\processed_data\MR"
mr_dirs_list = os.listdir(mr_basepath)

ct_basepath = r"G:\! project\2024- UnityDosePrediction\data\processed_data\CT"
ct_dirs_list = os.listdir(ct_basepath)

beam_basepath = r"G:\! project\2024- UnityDosePrediction\data\processed_data_revised_1\BEAM"
beam_dirs_list = os.listdir(beam_basepath)

oar_basepath = r"G:\! project\2024- UnityDosePrediction\data\processed_data\OAR"
oar_dirs_list = os.listdir(oar_basepath)

ptv_basepath = r"G:\! project\2024- UnityDosePrediction\data\processed_data\PTV"
ptv_dirs_list = os.listdir(ptv_basepath)

dose_basepath = r"G:\! project\2024- UnityDosePrediction\data\processed_data\DOSE"
dose_dirs_list = os.listdir(dose_basepath)

dcm_basepath = r"G:\! project\2024- UnityDosePrediction\data\dose_generation"
dcm_dirs_list = os.listdir(dcm_basepath)
dcm_plan_dirs_list = []
for i in range(len(ptv_dirs_list)):
    dcm_name = ptv_dirs_list[i].split("_")[1]
    nominate = []
    for j in range(len(dcm_dirs_list)):
        if dcm_name in dcm_dirs_list[j] :
            nominate.append(dcm_dirs_list[j])
    
    for nom in nominate:
        if "Dose" in nom:
            continue 
        if "StrctrSets" in nom:
            continue 
        else:
            the_dcm_name = nom 
    dcm_plan_dirs_list.append(the_dcm_name)

        


# ref_mr_itk = r"G:\! project\2024- UnityDosePrediction\data\processed_data\MR\P000_823500_mr.nii.gz"
res_spacing = np.array((3.0, 3.0, 3.0))

def MakeBeamOnehot(mr_itk, beam_basepath, beam_fol_dir, res_size, res_spacing,idx):
    beam_dirs = os.listdir(os.path.join(beam_basepath,beam_fol_dir))
    beam_list = []
    for beam_dir in beam_dirs:
        beam_itk = sitk.ReadImage(os.path.join(beam_basepath, beam_fol_dir, beam_dir))
        b_origin = beam_itk.GetOrigin()
        b_spacing = beam_itk.GetSpacing()
        b_direction = beam_itk.GetDirection()
        
        beam_arr = sitk.GetArrayFromImage(beam_itk)
        beam_arr[beam_arr > 1] = 1
        beam_itk = sitk.GetImageFromArray(beam_arr)
        beam_itk.SetSpacing(b_spacing)
        beam_itk.SetOrigin(b_origin)
        beam_itk.SetDirection(b_direction)
        beam_itk = resample(beam_itk, res_spacing, res_size, 0)
        beam_arr = sitk.GetArrayFromImage(beam_itk)
        beam_arr[beam_arr >=0.5] = 1
        beam_arr[beam_arr < 0.5] = 0
        beam_list.append(beam_arr.astype("uint16"))
    
    beam_all = np.squeeze(np.sum(np.array(beam_list), axis = 0))
    beam_all_itk = sitk.GetImageFromArray(beam_all)
    beam_all_itk.SetOrigin(mr_itk.GetOrigin())
    beam_all_itk.SetSpacing(mr_itk.GetSpacing())
    beam_all_itk.SetDirection(mr_itk.GetDirection())
    beam_savepath = os.path.join(r"C:\! Project\2024_unitydoseprediction\traindata", "imagesTr_BEAM")
    os.makedirs(beam_savepath, exist_ok=True)
    sitk.WriteImage(beam_all_itk, os.path.join(beam_savepath, "%s_BEAM_onehot.nii.gz" %(beam_fol_dir)))
    
    return None

def GetOARdict(mr_itk, oar_basepath, oar_fol_dir, res_size, res_spacing, idx):
    oar_name_dict = {"heart." : None, 
                     "esophagus." : None, 
                     "kidney_l." : None, 
                     "kidney_r." : None, 
                     "lung_l." : None, 
                     "lung_r." : None,
                     "spinalcord." : None,
                     "stomach." : None,
                     "liver." : None,
                     "bowel" : None}
    

    oar_dirs = os.listdir(os.path.join(oar_basepath, oar_fol_dir))
    for oar_dir in oar_dirs:
        for key, val in oar_name_dict.items():
            
            if key != "bowel":
                if key in oar_dir:
                    oar_itk = sitk.ReadImage(os.path.join(oar_basepath, oar_fol_dir, oar_dir))
                    oar_itk = resample(oar_itk, res_spacing, res_size, 0)
                    oar_arr = sitk.GetArrayFromImage(oar_itk)
                    oar_arr[oar_arr >=0.5] = 1
                    oar_arr[oar_arr < 0.5] = 0
                    
                    oar_name_dict[key] = oar_arr.astype("uint8")
            else:
                oar_itk = sitk.ReadImage(os.path.join(oar_basepath, oar_fol_dir, oar_dir))
                oar_itk = resample(oar_itk, res_spacing, res_size, 0)
                oar_arr = sitk.GetArrayFromImage(oar_itk)
                oar_arr[oar_arr >=0.5] = 1
                oar_arr[oar_arr < 0.5] = 0
                oar_arr = oar_arr.astype("uint8")
                if val is None:
                    
                    oar_name_dict[key] = [oar_arr]
                else:
                    oar_name_dict[key].append(oar_arr)
    
    npy_res_size  = np.array([res_size[2], res_size[1], res_size[0]])
    for key, val in oar_name_dict.items():
        if key != "bowel":
            if val is None:
                oar_name_dict[key] = np.zeros(npy_res_size).astype("uint8")
        else:
            if val is None:
                oar_name_dict[key] = np.zeros(npy_res_size).astype("uint8")
            else:
                temp_arr = np.sum(np.array(oar_name_dict[key]), axis = 0)
                temp_arr = np.squeeze(temp_arr)
                temp_arr[temp_arr > 1] = 1 
                oar_name_dict[key] = temp_arr.astype("uint8")
                
    oar_savepath = os.path.join(r"C:\! Project\2024_unitydoseprediction\traindata", "imagesTr_OARs")
    os.makedirs(oar_savepath , exist_ok=True)
    for key, val in oar_name_dict.items():
        oar_itk = sitk.GetImageFromArray(val)
        oar_itk.SetSpacing(mr_itk.GetSpacing())
        oar_itk.SetOrigin(mr_itk.GetOrigin())
        oar_itk.SetDirection(mr_itk.GetDirection())
        if key != "bowel":
            sitk.WriteImage(oar_itk, os.path.join(oar_savepath, "%s_%s.nii.gz" %(oar_fol_dir,key[:-1])))
        else:
            sitk.WriteImage(oar_itk, os.path.join(oar_savepath, "%s_%s.nii.gz" %(oar_fol_dir,key)))
    return None
def GetPTV(mr_itk,ptv_basepath, ptv_dir, dcm_basepath, dcm_plan_dir, res_size, res_spacing, idx):
    ds = pydicom.dcmread(os.path.join(dcm_basepath, dcm_plan_dir), force=True)
    prescription = float(ds.DoseReferenceSequence[0].TargetPrescriptionDose)
    ptv_itk = sitk.ReadImage(os.path.join(ptv_basepath, ptv_dir))
    ptv_itk = resample(ptv_itk, res_spacing, res_size, 0)
    ptv_arr = sitk.GetArrayFromImage(ptv_itk).astype("float64")
    ptv_arr[ptv_arr >= 0.5] = 1 
    ptv_arr[ptv_arr < 0.5] = 0
    ptv_arr *= prescription
    ptv_itk = sitk.GetImageFromArray(ptv_arr.astype("float32"))
    ptv_itk.SetSpacing(mr_itk.GetSpacing())
    ptv_itk.SetOrigin(mr_itk.GetOrigin())
    ptv_itk.SetDirection(mr_itk.GetDirection())
    ptv_savepath = os.path.join(r"C:\! Project\2024_unitydoseprediction\traindata", "imagesTr_PTV")
    os.makedirs(ptv_savepath , exist_ok=True)
    sitk.WriteImage(ptv_itk, os.path.join(ptv_savepath, "%s" %ptv_dir))
    
    return None

def GetDose(dose_basepath, dose_dir, res_size, res_spacing, idx):
    dose_itk = sitk.ReadImage(os.path.join(dose_basepath, dose_dir))
    dose_itk = resample(dose_itk, res_spacing, res_size, 0)
    dose_savepath = os.path.join(r"C:\! Project\2024_unitydoseprediction\traindata", "labelsTr_DOSE")
    os.makedirs(dose_savepath, exist_ok=True)
    sitk.WriteImage(dose_itk, os.path.join(dose_savepath, "%s" %dose_dir))
    return dose_itk

if __name__ == "__main__":
    
    idx_list = [idx for idx in range(len(mr_dirs_list))]
    for idx, mr_dir, beam_fol_dir, oar_fol_dir, ptv_dir, dose_dir, dcm_plan_dir in zip(
                                                            idx_list,
                                                            mr_dirs_list, 
                                                            beam_dirs_list, 
                                                            oar_dirs_list, 
                                                            ptv_dirs_list,
                                                            dose_dirs_list,
                                                            dcm_plan_dirs_list):
        
        print("CASE %d" %(idx))
        
        mr_itk = sitk.ReadImage(os.path.join(mr_basepath, mr_dir))
        
        res_size = np.int32(np.round(np.array(mr_itk.GetSize()) * np.array(mr_itk.GetSpacing()) / res_spacing))
        mr_itk = resample(mr_itk, res_spacing, res_size, 0)
        mr_arr = sitk.GetArrayFromImage(mr_itk)
        print("multiprocessing start!")
        result_queue = Queue()
        #make beam one hot
        process_beamonehot = Process(target=MakeBeamOnehot, args=(mr_itk,
                                                                    beam_basepath, 
                                                                  beam_fol_dir, 
                                                                  res_size, 
                                                                  res_spacing,
                                                                  idx))
        
        # Get oar
        process_getoardict = Process(target=GetOARdict, args=(
                                                            mr_itk, 
                                                            oar_basepath, 
                                                            oar_fol_dir, 
                                                            res_size, 
                                                            res_spacing,idx))

        process_getptv = Process(target=GetPTV, args=(mr_itk, 
                                                      ptv_basepath, 
                                                      ptv_dir, 
                                                      dcm_basepath,
                                                      dcm_plan_dir,
                                                      res_size, 
                                                      res_spacing,idx))
        
        process_getdose = Process(target=GetDose, args=(dose_basepath, 
                                                          dose_dir, 
                                                          res_size, 
                                                          res_spacing,idx))
        
        process_beamonehot.start()
        process_getoardict.start()
        process_getptv.start()
        process_getdose.start()
        
        process_beamonehot.join()
        process_getoardict.join()
        process_getptv.join()
        process_getdose.join()
        
        print("multiprocessing done!")
        # results = [result_queue.get(), result_queue.get(), result_queue.get(), result_queue.get()]
        # beam_all_itk = results[0]
        # oar_name_dict = results[1]
        # ptv_itk = results[2]
        # dose_itk = results[3]
        print('done')
        # get ptv
        
        
        print('saving...')
        mr_savepath = os.path.join(r"C:\! Project\2024_unitydoseprediction\traindata", "imagesTr_MR")
        os.makedirs(mr_savepath , exist_ok=True)
        
    
        sitk.WriteImage(mr_itk, os.path.join(mr_savepath, mr_dir))
        
        
        
        
            
        print("progress done... Case %d/%d" %(idx+1, len(mr_dirs_list)))
    
    