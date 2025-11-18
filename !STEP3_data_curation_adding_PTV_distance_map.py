import os 
import SimpleITK as sitk
import openpyxl 
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
# dis_map_p = ndimage.morphology.distance_transform_edt(mask, sampling=spacing)
# dis_map_n = ndimage.morphology.distance_transform_edt(1-mask, sampling=spacing)
# dis_map = (dis_map_p - dis_map_n) / 100

suffix_list = [
"_stomach.nii.gz",
"_bowel.nii.gz",
"_duodenum.nii.gz",
"_esophagus.nii.gz",
"_heart.nii.gz",
"_kidney_l.nii.gz",
"_kidney_r.nii.gz",
"_liver.nii.gz",
"_lung_l.nii.gz",
"_lung_r.nii.gz",
"_spinalcord.nii.gz"]

basepath = r"X:\!project\!2025_doseprediction\data\processed_data\traindata_128_4\imagesTr_HPTV"
im_basepath = r"X:\!project\!2025_doseprediction\data\processed_data\traindata_128_4\imagesTr_MR"
savepath = r"X:\!project\!2025_doseprediction\data\processed_data\traindata_128_4\imagesTr_HPTV_distance_map"
im_dirs_list = sorted(os.listdir(im_basepath))
ptv_dirs_list = sorted(os.listdir(basepath))

for idx, (im_dir, ptv_dir) in enumerate(zip(im_dirs_list, ptv_dirs_list)):
    
    ptv_itk = sitk.ReadImage(os.path.join(basepath, ptv_dir))
    ptv_arr = sitk.GetArrayFromImage(ptv_itk)
    ptv_arr[ptv_arr > 1] = 1
    if np.all(ptv_arr == 0):
        dis_map = np.zeros_like(ptv_arr)
        dis_map_itk = sitk.GetImageFromArray(dis_map)
        dis_map_itk.CopyInformation(ptv_itk)
    else:
        spacing = (ptv_itk.GetSpacing()[2], ptv_itk.GetSpacing()[1], ptv_itk.GetSpacing()[0])
        dis_map_p = ndimage.morphology.distance_transform_edt(ptv_arr, sampling=spacing)
        dis_map_n = ndimage.morphology.distance_transform_edt(1-ptv_arr, sampling=spacing)
        dis_map = (dis_map_p - dis_map_n) / 100
        dis_map_itk = sitk.GetImageFromArray(dis_map)
        dis_map_itk.CopyInformation(ptv_itk)
    sitk.WriteImage(dis_map_itk, os.path.join(r"X:\!project\!2025_doseprediction\data\processed_data\traindata_128_4\imagesTr_HPTV_distance_map", ptv_dir.split(".")[0] + "_distance.nii.gz"))
    