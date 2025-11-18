
import os
import json
from collections import OrderedDict



mode = "f0_test"
# if "r" in mode:
#     ct_fol_mode = "CECT_crop_flip"
# else:
ct_fol_mode = "image"

basepath = r"X:\!project\!2025_doseprediction\data\processed_data\traindata_128"
mr_basepath = os.path.join(basepath, "imagesTr_MR")
mr_dirs_list =os.listdir(mr_basepath)
beam_basepath = os.path.join(basepath, "imagesTr_BEAM")
beam_dirs_list = os.listdir(beam_basepath)
oar_basepath = os.path.join(basepath, "imagesTr_OARs")
hptv_basepath = os.path.join(basepath, "imagesTr_HPTV")
hptv_dirs_list = os.listdir(hptv_basepath)
lptv_basepath = os.path.join(basepath, "imagesTr_LPTV")
lptv_dirs_list = os.listdir(lptv_basepath)



la_basepath = os.path.join(basepath, "labelsTr_DOSE")
la_dir_list = os.listdir(la_basepath)




file_data = OrderedDict()
file_data["description"] = "challengedata"
file_data["labels"] = {"0" : "background", "1": "bbox"}
file_data["licence"] = "hand off!"
file_data["modality"] = {"0" : "CT"}
file_data["name"] = "segrap"
file_data["numTest"] = 20
file_data["numTraining"] = 50
file_data["reference"] = ""
file_data["release"] = "0.0"
file_data["tensorImageSize"] = "4D"

file_data["training"] = []
file_data["validation"] = []
file_data["test"] = []
oar_name_dict = {"heart" : None, 
                     "esophagus" : None, 
                     "kidney_l" : None, 
                     "kidney_r" : None, 
                     "lung_l" : None, 
                     "lung_r" : None,
                     "spinalcord" : None,
                     "stomach" : None,
                     "liver" : None,
                     "duodenum": None
                     }
val_list = [
"P001_900110" ,
"P002_1291099" ,
"P004_1920499" ,
"P010_4382437" ,
"P015_6187855" ,
"P038_9329948",
"P040_9349275",
"P041_9358134"]

test_list = [
"P025_6673577" ,
"P030_9027049" ,
"P033_9129435" ,
"P035_9244504" ,
"P039_9341591" ,
"P042_9365137" ,
"P047_9422096" ,
"P049_9429186" ,
"P054_9483358" ,
"P057_10079813"
]

for idx in range(len(mr_dirs_list)):   
    print(idx) 
    pnum = str(beam_dirs_list[idx].split("_")[0] + "_" +  beam_dirs_list[idx].split("_")[1])
    print(pnum)
    if pnum in test_list:
        continue
    
    image_list = ["./imagesTr_MR/"  + mr_dirs_list[idx],
                  "./imagesTr_BEAM/" + beam_dirs_list[idx],
                  "./imagesTr_HPTV/" + hptv_dirs_list[idx],
                  "./imagesTr_LPTV/" + lptv_dirs_list[idx]]
    
    for key in oar_name_dict.keys():
        image_list.append("./imagesTr_OARs/" + pnum + "_" + key + ".nii.gz")
    
    
    if  pnum in val_list:
        
        file_data["validation"].append({"image": image_list[:],
                                    "label": "./labelsTr_DOSE/"  + la_dir_list[idx]})
        
    else:
        file_data["training"].append({"image": image_list[:],
                                    "label": "./labelsTr_DOSE/"  + la_dir_list[idx]})
# test json

for idx in range(len(mr_dirs_list)):   
     
    pnum = str(beam_dirs_list[idx].split("_")[0] + "_" +  beam_dirs_list[idx].split("_")[1])
    
    image_list = ["./imagesTr_MR/"  + mr_dirs_list[idx],
                  "./imagesTr_BEAM/" + beam_dirs_list[idx],
                  "./imagesTr_HPTV/" + hptv_dirs_list[idx],
                  "./imagesTr_LPTV/" + lptv_dirs_list[idx]]
    if pnum in test_list:
        for key in oar_name_dict.keys():
            image_list.append("./imagesTr_OARs/" + pnum + "_" + key + ".nii.gz")
        file_data["test"].append({"image": image_list[:],
                                    "label": "./labelsTr_DOSE/"  + la_dir_list[idx]})
    
file_path = basepath + "/" + r"/dataset_%s" %( mode) + ".json"
with open(file_path, 'w', encoding='utf-8') as file:
    json.dump(file_data, file, indent=4)
