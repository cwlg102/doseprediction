import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import shutil
import tempfile
import nibabel as nib
import numpy as np
import time

from tqdm import tqdm
import torch
import functools
from torch.nn import init
from monai.losses import DiceCELoss
from monai.losses.ssim_loss import SSIMLoss
from monai.inferers import sliding_window_inference
from monai.transforms import (
    AsDiscrete,
    Compose,
    Resized,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandSpatialCropSamplesd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    EnsureTyped,
    RandGaussianNoise,
    RandGaussianSmooth,
    RandZoomd,
    RandFlip,
    RandRotate90,
    RandAdjustContrast,
    RandShiftIntensity,
    RandGibbsNoise,
    ScaleIntensity,
    RandSimulateLowResolutiond
)

from monai.config import print_config
from monai.metrics import MAEMetric
from monai.networks.nets import (SwinUNETR, UNETR, UNet, DynUNet, SegResNet)

from monai.data import (
    ThreadDataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
    set_track_meta,
)
import argparse
import torch
from torch import nn
from torch.optim.lr_scheduler import _LRScheduler

import math
import copy
from torch.nn import init
import functools
from torch.optim import lr_scheduler

import torch
import torch.nn as nn
import torch.nn.functional as F
class SingleConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding):
        super(SingleConv, self).__init__()

        self.single_conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, stride=stride, bias=True),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.single_conv(x)


class DenseConvolve(nn.Module):
    def __init__(self, in_ch, growth_rate=16, stride=(1, 1, 1)):
        super(DenseConvolve, self).__init__()

        self.single_conv = nn.Sequential(
            nn.Conv3d(in_ch, growth_rate, kernel_size=(3, 3, 3), padding=1, stride=stride, bias=True),
            nn.InstanceNorm3d(growth_rate, affine=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return torch.cat((self.single_conv(x), x), dim=1)


class DenseDownsample(nn.Module):
    def __init__(self, in_ch, growth_rate=16, stride=(2, 2, 2)):
        super(DenseDownsample, self).__init__()

        self.single_conv = nn.Sequential(
            nn.Conv3d(in_ch, growth_rate, kernel_size=(3, 3, 3), padding=1, stride=stride, bias=True),
            nn.InstanceNorm3d(growth_rate, affine=True),
            nn.ReLU(inplace=True)
        )

        self.pooling = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

    def forward(self, x):
        return torch.cat((self.single_conv(x), self.pooling(x)), dim=1)


class UNetUpsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UNetUpsample, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=(3, 3, 3), padding=1, stride=(1, 1, 1), bias=True),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=True)
        x = self.conv(x)
        return x


class Encoder(nn.Module):
    def __init__(self, in_ch, growth_rate=16):
        super(Encoder, self).__init__()
        self.encoder_1 = nn.Sequential(
            DenseConvolve(in_ch, growth_rate),
            DenseConvolve(in_ch + growth_rate, growth_rate),
        )
        self.encoder_2 = nn.Sequential(
            DenseDownsample(in_ch + 2 * growth_rate, growth_rate),
            DenseConvolve(in_ch + 3 * growth_rate, growth_rate),
            DenseConvolve(in_ch + 4 * growth_rate, growth_rate)
        )
        self.encoder_3 = nn.Sequential(
            DenseDownsample(in_ch + 5 * growth_rate, growth_rate),
            DenseConvolve(in_ch + 6 * growth_rate, growth_rate),
            DenseConvolve(in_ch + 7 * growth_rate, growth_rate)
        )
        self.encoder_4 = nn.Sequential(
            DenseDownsample(in_ch + 8 * growth_rate, growth_rate),
            DenseConvolve(in_ch + 9 * growth_rate, growth_rate),
            DenseConvolve(in_ch + 10 * growth_rate, growth_rate)
        )
        self.encoder_5 = nn.Sequential(
            DenseDownsample(in_ch + 11 * growth_rate, growth_rate),
            DenseConvolve(in_ch + 12 * growth_rate, growth_rate),
            DenseConvolve(in_ch + 13 * growth_rate, growth_rate),
            DenseConvolve(in_ch + 14 * growth_rate, growth_rate),
            DenseConvolve(in_ch + 15 * growth_rate, growth_rate)
        )

    def forward(self, x):
        out_encoder_1 = self.encoder_1(x)
        out_encoder_2 = self.encoder_2(out_encoder_1)
        out_encoder_3 = self.encoder_3(out_encoder_2)
        out_encoder_4 = self.encoder_4(out_encoder_3)
        out_encoder_5 = self.encoder_5(out_encoder_4)

        return [out_encoder_1, out_encoder_2, out_encoder_3, out_encoder_4, out_encoder_5]


class Decoder(nn.Module):
    def __init__(self, in_ch, growth_rate, upsample_chan, out_ch):
        super(Decoder, self).__init__()

        self.upconv_4 = UNetUpsample(in_ch + 16 * growth_rate, upsample_chan)
        self.decoder_conv_4 = nn.Sequential(
            SingleConv(in_ch + 11 * growth_rate + upsample_chan, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                       padding=1),
            SingleConv(256, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1)
        )
        self.upconv_3 = UNetUpsample(256, upsample_chan)
        self.decoder_conv_3 = nn.Sequential(
            SingleConv(in_ch + 8 * growth_rate + upsample_chan, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                       padding=1),
            SingleConv(128, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1)
        )
        self.upconv_2 = UNetUpsample(128, upsample_chan)
        self.decoder_conv_2 = nn.Sequential(
            SingleConv(in_ch + 5 * growth_rate + upsample_chan, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1),
            SingleConv(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1)
        )
        self.upconv_1 = UNetUpsample(64, upsample_chan)
        self.decoder_conv_1 = nn.Sequential(
            SingleConv(in_ch + 2 * growth_rate + upsample_chan, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1),
            SingleConv(32, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1)
        )

        self.final_conv = nn.Conv3d(32, out_ch, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=True)

    def forward(self, out_encoder):
        out_encoder_1, out_encoder_2, out_encoder_3, out_encoder_4, out_encoder_5 = out_encoder

        out_decoder_4 = self.decoder_conv_4(
            torch.cat((self.upconv_4(out_encoder_5), out_encoder_4), dim=1)
        )
        out_decoder_3 = self.decoder_conv_3(
            torch.cat((self.upconv_3(out_decoder_4), out_encoder_3), dim=1)
        )
        out_decoder_2 = self.decoder_conv_2(
            torch.cat((self.upconv_2(out_decoder_3), out_encoder_2), dim=1)
        )
        out_decoder_1 = self.decoder_conv_1(
            torch.cat((self.upconv_1(out_decoder_2), out_encoder_1), dim=1)
        )

        final_output = self.final_conv(out_decoder_1)
        return final_output


class HD_UNet(nn.Module):
    def __init__(self, in_ch, growth_rate, upsample_chan, out_ch):
        super(HD_UNet, self).__init__()
        self.encoder = Encoder(in_ch, growth_rate)
        self.decoder = Decoder(in_ch, growth_rate, upsample_chan, out_ch)

        # init
        self.initialize()

    @staticmethod
    def init_conv_IN(modules):
        for m in modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.InstanceNorm3d):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)

    def initialize(self):
        print('# random init encoder weight using nn.init.kaiming_uniform !')
        self.init_conv_IN(self.decoder.modules)
        print('# random init decoder weight using nn.init.kaiming_uniform !')
        self.init_conv_IN(self.encoder.modules)

    def forward(self, x):
        out_encoder = self.encoder(x)
        out_decoder = self.decoder(out_encoder)

        # Output is a list: [Output]
        return out_decoder


class Model(nn.Module):
    def __init__(self, in_ch, growth_rate, upsample_chan, out_ch):
        super(Model, self).__init__()

        self.model = HD_UNet(in_ch, growth_rate, upsample_chan, out_ch)

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":

    spatial_size_xyz = (96, 96, 64)
    
    

    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    OAR_nums_plus_one = 1
    patch_size = list(spatial_size_xyz)
    spacing = [3.0, 3.0, 3.0]

    # netG = Model(in_ch=16, growth_rate=16, upsample_chan=64, out_ch=1).to(device)
    netG = UNETR(in_channels=16, out_channels=1, img_size=spatial_size_xyz, feature_size=32).to(device)
    # netG = UNETR(in_channels=14, out_channels=1, img_size=spatial_size_xyz).to(device)
    netG.load_state_dict(torch.load(r"X:\!project\!2025_doseprediction\ckpt\model_hdunet_exp22_gan\doseprediction_modelG_100_0.0036271.pth")["netG_state_dict"])
    netG.eval()
    
    basepath = r"X:\!project\!2025_doseprediction\data\processed_data\traindata_128_2"
    BEAM_path = os.path.join(basepath, "imagesTr_BEAM")
    HPTV_path = os.path.join(basepath, "imagesTr_HPTV")
    LPTV_path = os.path.join(basepath, "imagesTr_LPTV")
    MR_path = os.path.join(basepath, "imagesTr_MR")
    OAR_path = os.path.join(basepath , "imagesTr_OARs")
    site_path = os.path.join(basepath, "imagesTr_additional_info_site")
    treat_path = os.path.join(basepath, "imagesTr_additional_info_treatment")

    
    BEAM_dirs_list = sorted(os.listdir(BEAM_path))
    HPTV_dirs_list = sorted(os.listdir(HPTV_path))
    LPTV_dirs_list = sorted(os.listdir(LPTV_path))
    MR_dirs_list = sorted(os.listdir(MR_path))
    OAR_dirs_list = sorted(os.listdir(OAR_path))
    site_dirs_list = sorted(os.listdir(site_path))
    treat_dirs_list = sorted(os.listdir(treat_path))

    oar_list = ["heart", "esophagus", "kidney_l", "kidney_r", "lung_l", "lung_r", "spinalcord", "stomach", "liver", "duodenum"]
    scale_intensity = ScaleIntensity(minv=0, maxv=1, channel_wise=True)
    mrn_list = ["P025", "P030", "P033", "P035", "P039", "P042", "P047", "P049", "P054", "P057"]
    import SimpleITK as sitk 
    savepath = r"X:\!project\!2025_doseprediction\results\exp22"
    os.makedirs(savepath, exist_ok=True)
    for idx, (mr_dir, beam_dir, hptv_dir, lptv_dir, oar_dir, site_dir, treat_dir) in enumerate(zip(MR_dirs_list, BEAM_dirs_list, HPTV_dirs_list, LPTV_dirs_list, OAR_dirs_list, site_dirs_list, treat_dirs_list)):

        if mr_dir.split("_")[0] in mrn_list:
            pass
        else:
            continue
        mr_itk = sitk.ReadImage(os.path.join(MR_path, mr_dir))
        mr_arr = sitk.GetArrayFromImage(mr_itk)
        beam_itk = sitk.ReadImage(os.path.join(BEAM_path, beam_dir))
        beam_arr = sitk.GetArrayFromImage(beam_itk)
        hptv_itk = sitk.ReadImage(os.path.join(HPTV_path, hptv_dir))
        hptv_arr = sitk.GetArrayFromImage(hptv_itk)
        lptv_itk = sitk.ReadImage(os.path.join(LPTV_path, lptv_dir))
        lptv_arr = sitk.GetArrayFromImage(lptv_itk)
        site_itk = sitk.ReadImage(os.path.join(site_path, site_dir))
        site_arr = sitk.GetArrayFromImage(site_itk)
        treat_itk = sitk.ReadImage(os.path.join(treat_path, treat_dir))
        treat_arr = sitk.GetArrayFromImage(treat_itk)

        mr_arr = torch.tensor(mr_arr[np.newaxis, np.newaxis, ...].astype("float32"))
        beam_arr = torch.tensor(beam_arr[np.newaxis, np.newaxis, ...].astype("float32"))
        hptv_arr = torch.tensor(hptv_arr[np.newaxis, np.newaxis, ...].astype("float32"))
        lptv_arr = torch.tensor(lptv_arr[np.newaxis, np.newaxis, ...].astype("float32"))
        treat_arr = torch.tensor(treat_arr[np.newaxis, np.newaxis, ...].astype("float32"))
        site_arr = torch.tensor(site_arr[np.newaxis, np.newaxis, ...].astype("float32"))
        mr_arr = scale_intensity(mr_arr)
        beam_arr = scale_intensity(beam_arr)
        hptv_arr /= 100 
        lptv_arr /= 100
        realA = torch.cat([mr_arr, beam_arr, hptv_arr, lptv_arr, site_arr, treat_arr], dim=1)
        for oar in oar_list:
            oar_itk = sitk.ReadImage(os.path.join(OAR_path, mr_dir.split(".")[0][:-3] + "_" + oar + ".nii.gz"))
            oar_arr = sitk.GetArrayFromImage(oar_itk)
            oar_arr = torch.tensor(oar_arr[np.newaxis, np.newaxis, ...].astype("float32"))
            realA = torch.cat([realA, oar_arr], dim=1)
        realA = realA.permute(0, 1, 4, 3, 2)
            
        # tx =  torch.reshape(realA[:, 0, :, : ,:].clone().detach(), (realA.shape[0], 1, realA.shape[2], realA.shape[3], realA.shape[4]))
        # beam = torch.reshape(realA[:, 1, :, : ,:].clone().detach(), (realA.shape[0], 1, realA.shape[2], realA.shape[3], realA.shape[4]))
        # hptv = torch.reshape(realA[:, 2, :, : ,:].clone().detach(), (realA.shape[0], 1, realA.shape[2], realA.shape[3], realA.shape[4]))
        # lptv = torch.reshape(realA[:, 3, :, : ,:].clone().detach(), (realA.shape[0], 1, realA.shape[2], realA.shape[3], realA.shape[4]))

        # tx = randcontrast(tx)
        # tx = rgnoise(tx)
        # tx = randgibbs(tx)
        # beam = scale_intensity(beam)
        # beam = randcontrast(beam)
        # beam = rgnoise(beam)
        # hptv /= 100
        # lptv /= 100
        # realA = torch.cat((tx, beam, hptv, lptv, realA[:, 4:, :, :, :]), 1)
        # realB = scale_intensity(realB)
        # realB /= 100
        realA = realA.to(device)
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                fakeB = sliding_window_inference(realA, spatial_size_xyz, 4, netG, 0.5, mode="gaussian")
        
        fakeB *= 100
        output = fakeB.clone().detach().cpu().numpy()
        output = np.squeeze(output)
        output = np.transpose(output, (2, 1, 0))
        out_itk = sitk.GetImageFromArray(output)
        out_itk.CopyInformation(mr_itk)
        
        sitk.WriteImage(out_itk, os.path.join(savepath, mr_dir.split(".")[0] + "_dose_output.nii.gz"))
        
    # original_nib_path_list = os.listdir(original_nib_path)
    