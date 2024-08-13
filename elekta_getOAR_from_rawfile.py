import os
import re
import random
import numpy as np
import pydicom
from skimage.draw import line as lin
from skimage.draw import polygon
import SimpleITK as sitk

def main():
    done_path = r"G:\! project\2024- UnityDosePrediction\data\dose_generation"
    dose_dirs_list = os.listdir(done_path)
    iwantthis = []
    
    for idx in range(0, len(dose_dirs_list),3):
        iwantthis.append(dose_dirs_list[idx].split("_")[0])
        
    ct_path = r"G:\gangnam_data"
    ct_savepath = r"G:\! project\2024- UnityDosePrediction\data\processed_data_ctwithOAR\CT"
    label_savepath = r"G:\! project\2024- UnityDosePrediction\data\processed_data_ctwithOAR\OAR"
    ct_list = os.listdir(ct_path)
    idx = 0
    for i in range(len(iwantthis)):
        pt_dir = os.path.join(ct_path, iwantthis[i]+"_all_fx", "1~CT1")
        contour_info_from_elekta(pt_dir, i, ct_savepath, label_savepath)
        idx += 1
        
        

def contour_info_from_elekta(pt_dir, idx,  ct_savepath, label_savepath):
    contournames_path = os.path.join(pt_dir, "contournames")
    contour_dict = get_contournames_info(contournames_path)
    # if i_want_t   his_oar in contour_dict:
    #     target_val = contour_dict[i_want_this_oar]
    
            
    # contour_dict = {"test" : 9}
    # color_dict = {}
    # for key, val in contour_dict.items():
    #     color_dict[int(val)] = rand_color_generator()
    
    dcm_fol_path = os.path.join(pt_dir, "DCMData")
    
    file_list = os.listdir(pt_dir)


    wc_file_sort_list = []
    for i in range(len(file_list)):
        if file_list[i][-2:] == "WC":
            
            wc_file_path = os.path.join(pt_dir, file_list[i])
            with open(wc_file_path, "r") as wc_file:
                wc_file_readlines = wc_file.readlines()
            
            wc_file_z_coord = re.sub("\n", "", wc_file_readlines[5].split(",")[2])
            wc_file_z_coord=  float(wc_file_z_coord)
            wc_file_sort_list.append((file_list[i], wc_file_z_coord))
            
    wc_file_sort_list.sort(key=lambda x: x[1])
    
    dcm_list = ct_dcm_data(dcm_fol_path)
    assert len(dcm_list) == len(wc_file_sort_list), "match file size"
    
    file_length = len(dcm_list)
    inf_origin_dcm = dcm_list[0][0]
    inf_z = dcm_list[0][1]
    sup_z = dcm_list[-1][1]
    
    origin_coordinate = np.array(inf_origin_dcm.ImagePositionPatient)
    Z_SPACING = (sup_z - inf_z)/(file_length-1)
    X_SPACING, Y_SPACING = inf_origin_dcm.PixelSpacing[0], inf_origin_dcm.PixelSpacing[1]
    
    
    ct_arr = get_dcm_3d_arr(dcm_list)
    ##visualization##
    # vi_arr = np.copy(ct_arr) 
    # win_min = -160
    # win_max = 350
    
    # vi_arr[vi_arr > win_max] = win_max
    # vi_arr[vi_arr < win_min] = win_min
    # vi_arr = 255 * (vi_arr - win_min)/(win_max - win_min)
    # vi_arr = vi_arr.astype('uint8')
    # vi_arr = np.stack((vi_arr, ) * 3 , axis = -1)
    
    contour_reverse_dict = {v:k for k, v in contour_dict.items()}
    
    
    # la_arr = np.zeros_like(ct_arr).astype("uint8")
    contour_arr_dict = {}
    for key, val in contour_dict.items():
        contour_arr_dict[key] = np.zeros_like(ct_arr).astype("uint8")
        
    for i in range(len(dcm_list)):
        coord_dict = transfer_pixelspace_contouring(dcm_list[i][0], os.path.join(pt_dir, wc_file_sort_list[i][0]), contour_dict)
        
        for key, val in coord_dict.items():
            if val is None:
                continue
            else:
                for co_arr in val:
                    
                    # color = color_dict[key]
                    # poly_r = []
                    # poly_c = []
                    # for j in range(len(val)-1):
                    #     x1 = val[j][0]
                    #     y1 = val[j][1]
                    #     x2 = val[j+1][0]
                    #     y2 = val[j+1][1]
                    #     ry, cx = lin(y1, x1, y2, x2)
                    #     poly_r.append(y1)
                    #     poly_c.append(x1)
                    #     la_arr[i, ry, cx] = 1
                    # poly_r.append(y2)
                    # poly_c.append(x2)
                    # poly_r.append(val[0][1])
                    # poly_c.append(val[0][0])

                    # ry,cx = lin(y2, x2, val[0][1], val[0][0])
                    # la_arr[i, ry, cx] = 1
                    # rr, cc = polygon(poly_r, poly_c)
                    # la_arr[i, rr, cc] = 1

                    co_arr = np.transpose(co_arr)
                    cc, rr = polygon(co_arr[0], co_arr[1])
                    contour_arr_dict[contour_reverse_dict[str(key)]][i, rr, cc] = 1
                    # la_arr[i, rr, cc] = 1
            
            
    ct_arr = ct_arr.astype("int32")

    print(np.max(ct_arr), np.min(ct_arr))

    # np.save(ct_arr, os.path.join(r"D:\!JUNG_newdata_T23D_mr\abdomenCT1_npy", "P%03d_CT.npy" %idx))        
    # np.save(la_arr, os.path.join(r"D:\!JUNG_newdata_T23D_mr\abdomenCT1_ctr_npy", "P%03d_RTST.npy" %idx))
    ct_sitk = sitk.GetImageFromArray(ct_arr)
    ct_sitk.SetOrigin(origin_coordinate)
    ct_sitk.SetSpacing((X_SPACING, Y_SPACING, Z_SPACING))
    ct_sitk.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    sitk.WriteImage(ct_sitk, os.path.join(ct_savepath, "P%03d_CT.nii.gz" %idx))        
    
    os.makedirs(os.path.join(label_savepath, "P%03d" %(idx)), exist_ok=True)
    for key, val in contour_arr_dict.items():
        la_sitk = sitk.GetImageFromArray(val)
        la_sitk.SetOrigin(origin_coordinate)
        la_sitk.SetSpacing((X_SPACING, Y_SPACING, Z_SPACING))
        la_sitk.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
        try:
            sitk.WriteImage(la_sitk, os.path.join(label_savepath, "P%03d" %(idx),"P%03d_%s.nii.gz" %(idx, key)))
        except:
            continue
def get_contournames_info(contournames_path):
    with open(contournames_path, "r") as contournames:
        cnames_line_list = contournames.readlines()
    
    contour_dict = {}
    for idx, line in enumerate(cnames_line_list):
        line_str = line
        if "\n" in line_str:
            line_str = re.sub("\n", "", line_str)
        
        if line_str.isspace(): # space
            continue 
        elif line_str == "": # null
            continue
        
        number_det = re.findall(r"\d+", line_str) #Zn case except ex) Z8, Z1 ...
        check = 0
        if len(number_det) > 0:
            pass
        elif "gtv" in line_str.lower():
            check = 1
        else:
            if   line_str.lower() == "imported":
                pass
            elif line_str.lower() == "mrlcouch":
                pass
            elif line_str.lower() == "study ct":
                pass
            elif line_str.lower() == "mm":
                pass
            elif line_str.lower() == "patient":
                pass
            elif line_str.lower() == "general":
                pass
            elif line_str.lower() == "isocenter":
                pass
            else:
                check = 1
                
        
        if check == 1 and idx != len(cnames_line_list) -1:
            next_line = cnames_line_list[idx+1]
            contour_dict[line_str.lower()] = next_line.split(",")[0]
    return contour_dict

    

def transfer_pixelspace_contouring(dcm, wc_file_path, contour_dict):
    with open(wc_file_path, "r") as wc_file:
        wc_file_readlines = wc_file.readlines()
    #start index is 7
    idx = 7
    # translation_arr = np.array([float(wc_file_readlines[4].split(",")[0]), float(wc_file_readlines[4].split(",")[1])])
    # elekta_orientation_arr = np.array([float(wc_file_readlines[6].split(",")[0:3]), float(wc_file_readlines[6].split(",")[3:6])])
    
    origin_arr = np.array(dcm.ImagePositionPatient[:2])
    
    
    # spacing = dcm.PixelSpacing
    spacing_arr = np.array(dcm.PixelSpacing)
    
    
    
    coord_dict = {}
    for key, val in contour_dict.items():
        coord_dict[int(val)] = []
    
    while idx < len(wc_file_readlines):
        ctr_meta_coord = wc_file_readlines[idx]
        ctr_meta = wc_file_readlines[idx+1]
        # print(ctr_meta_coord)
        # print(ctr_meta)
        if "\n" in ctr_meta_coord:
            ctr_meta_coord = re.sub("\n", "", ctr_meta_coord)
            ctr_meta = re.sub("\n", "", ctr_meta)
        
       
        if "isocenter" == str(ctr_meta_coord):
            break
        
        ctr_coord_amount = int(ctr_meta_coord)
        ctr_target = int(ctr_meta)
        
        if ctr_coord_amount == 0: #NEED TO CHECK
            if wc_file_readlines[idx+3] == "Bart" and wc_file_readlines[idx+4] != "":
                idx = idx + 6
                continue
            else:
                break
        
        ctr_coord_start_idx = idx + 2
        
        # 5 set unit
        length = ctr_coord_amount // 5
        if ctr_coord_amount % 5 != 0:
            length += 1
        
        if ctr_target not in coord_dict:
            idx = idx + length + 2
            continue
        
        
        ctr_coord_list = []
        for ctr_line_idx in range(ctr_coord_start_idx, ctr_coord_start_idx + length):
            ctr_line =re.sub("\n", "",  wc_file_readlines[ctr_line_idx])
            ctr_line_list = ctr_line.split(",")
            temp_ctr_coord_list = [float(i) for i in ctr_line_list]
            ctr_coord_list.extend(temp_ctr_coord_list)
        ctr_coord_arr = np.array(ctr_coord_list)
        ctr_coord_arr = np.reshape(ctr_coord_arr, (len(ctr_coord_arr)//2, 2))
        #all???
        # if elekta_orientation_arr[0][1] == 1.0:
        ctr_coord_arr[:, 1] *= -1
        
        ctr_pixcd_arr = (ctr_coord_arr - origin_arr) /spacing_arr 
        
        ctr_pixcd_arr_int = ctr_pixcd_arr.astype("int32")
        ctr_pixcd_arr_int = ctr_pixcd_arr_int.astype("float64")
        ctr_pixcd_arr_alpha = ctr_pixcd_arr - ctr_pixcd_arr_int
        ctr_pixcd_arr_alpha[ctr_pixcd_arr_alpha >= 0.5] = 1
        ctr_pixcd_arr_alpha[ctr_pixcd_arr_alpha <0.5] = 0
        ctr_pixcd_arr_int += ctr_pixcd_arr_alpha
        ctr_pixcd_arr_int = ctr_pixcd_arr_int.astype("int32")
        coord_dict[ctr_target].append(ctr_pixcd_arr_int)
        
        idx = idx + length + 2
    
    return coord_dict

def ct_dcm_data(dcm_dir):
    dcm_fol_list = os.listdir(dcm_dir)
    dcm_list = []
    for idx, dcm_path in enumerate(dcm_fol_list):
        dcm_path_ = os.path.join(dcm_dir, dcm_path)
        dcm = pydicom.dcmread(dcm_path_, force=True)
        dcm.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
        dcm_list.append((dcm, dcm.ImagePositionPatient[2]))
    dcm_list.sort(key=lambda x:x[1])
    
    return dcm_list

def get_dcm_3d_arr(dcm_list):
    arr_list = []
    for i, dcm in enumerate(dcm_list):
        dcm_slice = dcm[0].pixel_array * dcm[0].RescaleSlope + dcm[0].RescaleIntercept
        arr_list.append(dcm_slice)
    dcm_3d_arr = np.array(arr_list)
    return dcm_3d_arr

main()
