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

basepath = r"X:\!project\!2025_doseprediction\data\processed_data\traindata_128_4\imagesTr_OARs"
im_basepath = r"X:\!project\!2025_doseprediction\data\processed_data\traindata_128_4\imagesTr_MR"
savepath = r"X:\!project\!2025_doseprediction\data\processed_data\traindata_128_4\imagesTr_OARs_distance_map"
im_dirs_list = sorted(os.listdir(im_basepath))

for idx, im_dir in enumerate(im_dirs_list):
    
    prefix = im_dir.split("_")[0] + "_" + im_dir.split("_")[1] 
    for jdx, oar_dir in enumerate(suffix_list):
        oar_itk = sitk.ReadImage(os.path.join(basepath, prefix + oar_dir))
        oar_arr = sitk.GetArrayFromImage(oar_itk)
        if np.all(oar_arr  == 0):
            dis_map = np.zeros_like(oar_arr)
            dis_map_itk = sitk.GetImageFromArray(dis_map)
            dis_map_itk.CopyInformation(oar_itk)
            # sitk.WriteImage(dis_map_itk, os.path.join(r"X:\!project\!2025_doseprediction\data\processed_data\traindata_128_4\imagesTr_distance_map_OARs", prefix + oar_dir.split(".")[0] + "_distance.nii.gz"))
        else:
            spacing = (oar_itk.GetSpacing()[2], oar_itk.GetSpacing()[1], oar_itk.GetSpacing()[0])
            dis_map_p = ndimage.morphology.distance_transform_edt(oar_arr, sampling=spacing)
            dis_map_n = ndimage.morphology.distance_transform_edt(1-oar_arr, sampling=spacing)
            dis_map = (dis_map_p - dis_map_n) / 100
            dis_map_itk = sitk.GetImageFromArray(dis_map)
            dis_map_itk.CopyInformation(oar_itk)
        sitk.WriteImage(dis_map_itk, os.path.join(r"X:\!project\!2025_doseprediction\data\processed_data\traindata_128_4\imagesTr_OARs_distance_map", prefix + oar_dir.split(".")[0] + "_distance.nii.gz"))
    