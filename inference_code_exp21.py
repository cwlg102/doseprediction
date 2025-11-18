"""
HD‑UNet‑fusion 수동 패치 추론 스크립트
===================================================
* Sliding‑window 대신 사용자가 원하는 "i, j, k 이동" 방식.
* 패치 크기와 overlap 비율을 지정하면 stride = patch*(1‑overlap) 로 자동 계산.
* 각 패치 결과를 output 배열에 누적 합산 → weight 배열(동일 크기)에 1씩 더함 → 마지막에 element‑wise 나눗셈으로 평균.
* 네트워크, 전처리, 채널 구성은 exp17 학습 코드와 동일 (realA_img 16ch, realA_dis 16ch).
* 필요 시 Gaussian weight kernel로 변경 가능(코드 주석 참고).
"""

import os
from typing import List
import numpy as np
import torch
import SimpleITK as sitk
from monai.transforms import ScaleIntensity
# from network import Model  # exp17 네트워크 정의 파일을 network.py 로 저장했다고 가정
import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from torch.optim import lr_scheduler
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

# class FusionLayer(nn.Module):
#     def __init__(self, c_im, c_dis, c_out):
#         super().__init__()
#         self.proj_im  = nn.Conv3d(c_im,  c_out, kernel_size=1, bias=False)
#         self.proj_dis = nn.Conv3d(c_dis, c_out, kernel_size=1, bias=False)

#     def forward(self, f_im, f_dis, mode="mul"):
#         f_im  = self.proj_im(f_im)
#         f_dis = self.proj_dis(f_dis)
#         if mode == "mul":
#             return f_im * f_dis          # element-wise
#         else:                            # concat+conv 방식
#             return torch.cat([f_im, f_dis], dim=1)

class HD_UNet_fusion(nn.Module):
    def __init__(self, in_ch, in_ch_dis, growth_rate, upsample_chan, out_ch):
        super(HD_UNet_fusion, self).__init__()
        self.encoder_im = Encoder(in_ch, growth_rate)
        self.encoder_dis = Encoder(in_ch_dis, growth_rate)
        # self.proj_dis = nn.Conv3d(44, 48, kernel_size=1, bias=False)
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
        self.init_conv_IN(self.encoder_im.modules)
        self.init_conv_IN(self.encoder_dis.modules)
    def forward(self, x, dis):
        out_encoder_im = self.encoder_im(x)
        out_encoder_dis = self.encoder_dis(dis)
        out_encoder = []
        for i in range(5):
            out_encoder.append(out_encoder_im[i] * out_encoder_dis[i])
        # out_encoder = out_encoder_im * out_encoder_dis
        out_decoder = self.decoder(out_encoder)

        # Output is a list: [Output]
        return out_decoder


class Model(nn.Module):
    def __init__(self, in_ch, in_ch_dis, growth_rate, upsample_chan, out_ch):
        super(Model, self).__init__()

        self.model = HD_UNet_fusion(in_ch, in_ch_dis, growth_rate, upsample_chan, out_ch)

    def forward(self, x, dis):
        return self.model(x, dis)

class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm3d, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv3d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)
    
# class PolyLRScheduler(_LRScheduler):
#     def __init__(self, optimizer, initial_lr: float, max_steps: int, exponent: float = 0.9, current_step: int = None):
#         self.optimizer = optimizer
#         self.initial_lr = initial_lr
#         self.max_steps = max_steps
#         self.exponent = exponent
#         self.ctr = 0
#         super().__init__(optimizer, current_step if current_step is not None else -1, False)

#     def step(self, current_step=None):
#         if current_step is None or current_step == -1:
#             current_step = self.ctr
#             self.ctr += 1

#         new_lr = self.initial_lr * (1 - current_step / self.max_steps) ** self.exponent
#         for param_group in self.optimizer.param_groups:
#             param_group['lr'] = new_lr
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)
    
norm_layer = functools.partial(nn.InstanceNorm3d, affine=False, track_running_stats=True)

import torch
import torch.nn.functional as F

def differentiable_dvh(dose, mask, dose_bins, beta=10.0):
    """
    dose: (B, 1, D, H, W)
    mask: (B, 1, D, H, W)
    dose_bins: 1D tensor of dose thresholds (nt,)
    beta: steepness of sigmoid approximation
    Returns: gDVH: (B, nt)
    """
    B = dose.shape[0]
    nt = dose_bins.shape[0]

    # Expand shapes to broadcast
    dose = dose.view(B, -1)                      # (B, N)
    mask = mask.view(B, -1)                      # (B, N)
    dose_bins = dose_bins.view(1, nt, 1)         # (1, nt, 1)
    dose = dose.unsqueeze(1)                     # (B, 1, N)
    mask = mask.unsqueeze(1)                     # (B, 1, N)

    # Differentiable volume >= dose_bin using sigmoid approximation
    v = torch.sigmoid(beta * (dose - dose_bins)) * mask   # (B, nt, N)
    v = v.sum(dim=2) / (mask.sum(dim=2) + 1e-6)          # Normalize by total voxels in mask -> (B, nt)
    return v

def dvh_loss(pred_dose, true_dose, structure_mask, n_bins=100, max_dose=80.0, beta=10.0):
    """
    pred_dose, true_dose: (B, 1, D, H, W)
    structure_mask: (B, 1, D, H, W)
    Returns: scalar loss
    """
    dose_bins = torch.linspace(0, max_dose, steps=n_bins, device=pred_dose.device)

    pred_dvh = differentiable_dvh(pred_dose, structure_mask, dose_bins, beta)
    true_dvh = differentiable_dvh(true_dose, structure_mask, dose_bins, beta)

    return F.mse_loss(pred_dvh, true_dvh)

class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm3d, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv3d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)

# --------------------------- 설정 ---------------------------
ROOT = r"X:\!project\!2025_doseprediction\data\processed_data\traindata_128_4"
CKPT = r"X:\!project\!2025_doseprediction\ckpt\model_hdunet_exp21_gan\doseprediction_modelG_100_.pth"
SAVE_DIR = r"X:\!project\!2025_doseprediction\results\exp21_infer_manualpatch"
PATCH_SIZE = (96, 96, 64)   # (Dx, Dy, Dz)
OVERLAP = 0.5               # 0~<1
SW_BATCH = 4                # GPU batch for patch processing
PATIENT_IDS = ["P025_6673577", "P030_9027049", "P033_9129435", "P035_9244504", "P039_9341591", "P042_9365137", "P047_9422096", "P049_9429186", "P054_9483358", "P057_10079813"]

# ------------------------------------------------------------
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
scale_intensity = ScaleIntensity(minv=0, maxv=1, channel_wise=True)

# ---------- 경로 정의 ----------
paths = {
    "MR": os.path.join(ROOT, "imagesTr_MR"),
    "BEAM": os.path.join(ROOT, "imagesTr_BEAM"),
    "HPTV": os.path.join(ROOT, "imagesTr_HPTV"),
    "LPTV": os.path.join(ROOT, "imagesTr_LPTV"),
    "SITE": os.path.join(ROOT, "imagesTr_additional_info_site"),
    "TREAT": os.path.join(ROOT, "imagesTr_additional_info_treatment"),
    "OAR": os.path.join(ROOT, "imagesTr_OARs"),
    "HPTV_DIST": os.path.join(ROOT, "imagesTr_HPTV_distance_map"),
    "LPTV_DIST": os.path.join(ROOT, "imagesTr_LPTV_distance_map"),
    "OAR_DIST": os.path.join(ROOT, "imagesTr_OARs_distance_map"),
}

oar_list = [
    "heart", "esophagus", "kidney_l", "kidney_r", "lung_l",
    "lung_r", "spinalcord", "stomach", "liver", "duodenum",
]

# ---------- 네트워크 ----------
netG = Model(in_ch=16, in_ch_dis=16, growth_rate=16, upsample_chan=64, out_ch=1).to(DEVICE)
state = torch.load(CKPT, map_location=DEVICE)
netG.load_state_dict(state["netG_state_dict"])
netG.eval()

def read_nii(path: str):
    itk = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(itk)         # [z,y,x]
    t = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    t = t.permute(0, 1, 4, 3, 2)              # [1,1,x,y,z] → [1,1,D,H,W]
    return t, itk

# ---------- 패치 grid ---------

def build_grid(dim: int, patch: int, overlap: float) -> List[int]:
    stride = max(1, int(patch * (1 - overlap)))
    coords = list(range(0, dim - patch + 1, stride))
    if coords[-1] != dim - patch:
        coords.append(dim - patch)
    return coords

# ---------- 패치 추론 ----------

def patch_inference(img: torch.Tensor,
                    dis: torch.Tensor,
                    model: torch.nn.Module,
                    patch_size: tuple,
                    overlap: float = 0.5,
                    batch: int = 4):
    """img/dis: [B, C, D, H, W] (B==1). returns [1,1,D,H,W]"""
    B, C, D, H, W = img.shape
    px, py, pz = patch_size
    xs = build_grid(D, px, overlap)
    ys = build_grid(H, py, overlap)
    zs = build_grid(W, pz, overlap)

    output = torch.zeros((1, 1, D, H, W), dtype=torch.float32, device=img.device)
    weight = torch.zeros_like(output)

    patches_img, patches_dis, indices = [], [], []

    for x in xs:
        for y in ys:
            for z in zs:
                patch_img = img[:, :, x:x+px, y:y+py, z:z+pz]
                patch_dis = dis[:, :, x:x+px, y:y+py, z:z+pz]
                patches_img.append(patch_img)
                patches_dis.append(patch_dis)
                indices.append((x, y, z))
                # 배치 처리
                if len(patches_img) == batch:
                    _infer_and_accumulate(patches_img, patches_dis, indices, model, output, weight)
                    patches_img, patches_dis, indices = [], [], []

    # 남은 패치 처리
    if patches_img:
        _infer_and_accumulate(patches_img, patches_dis, indices, model, output, weight)

    # weight>0 인 위치만 나눔
    output = output / torch.clamp(weight, min=1e-6)
    return output


def _infer_and_accumulate(p_img, p_dis, idx_list, model, out_tensor, w_tensor):
    """내부 util: 패치 리스트 추론 후 out/weight 누적"""
    batch_img = torch.cat(p_img, dim=0)
    batch_dis = torch.cat(p_dis, dim=0)
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=DEVICE.type == 'cuda'):
            pred = model(batch_img, batch_dis) * 100.0          # scale back 0‑100
    # accumulate
    for n, (x, y, z) in enumerate(idx_list):
        px, py, pz = pred[n].shape[-3:]
        out_tensor[:, :, x:x+px, y:y+py, z:z+pz] += pred[n:n+1]
        w_tensor[:, :, x:x+px, y:y+py, z:z+pz] += 1.0  # uniform weight

# ---------------- 메인 루프 ----------------
os.makedirs(SAVE_DIR, exist_ok=True)

for pid in PATIENT_IDS:
    print(f"\n▶ {pid} 시작")
    # ----- 채널 로드 -----
    mr, itk_ref = read_nii(os.path.join(paths["MR"], f"{pid}_mr.nii.gz"))
    beam, _ = read_nii(os.path.join(paths["BEAM"], f"{pid}_BEAM_onehot.nii.gz"))
    hptv, _ = read_nii(os.path.join(paths["HPTV"], f"{pid}_hptv.nii.gz"))
    lptv, _ = read_nii(os.path.join(paths["LPTV"], f"{pid}_lptv.nii.gz"))
    site, _ = read_nii(os.path.join(paths["SITE"], f"{pid}_site.nii.gz"))
    treat, _ = read_nii(os.path.join(paths["TREAT"], f"{pid}_treat.nii.gz"))

    oars = [read_nii(os.path.join(paths["OAR"], f"{pid}_{oar}.nii.gz"))[0] for oar in oar_list]
    hptv_dist, _ = read_nii(os.path.join(paths["HPTV_DIST"], f"{pid}_hptv_distance.nii.gz"))
    lptv_dist, _ = read_nii(os.path.join(paths["LPTV_DIST"], f"{pid}_lptv_distance.nii.gz"))
    oar_dists = [read_nii(os.path.join(paths["OAR_DIST"], f"{pid}_{oar}_distance.nii.gz"))[0] for oar in oar_list]

    # ----- 전처리 -----
    mr = scale_intensity(mr)
    beam = scale_intensity(beam)
    hptv = hptv / 100.0
    lptv = lptv / 100.0
    hptv_npy = hptv.cpu().numpy()
    hptv_npy = np.squeeze(hptv_npy)
    print(np.max(hptv_npy*100))
    realA_img = torch.cat([mr, beam, hptv, lptv, site, treat, *oars], dim=1).to(DEVICE)     # 16ch
    realA_dis = torch.cat([mr, beam, hptv, lptv, hptv_dist, lptv_dist, *oar_dists], dim=1).to(DEVICE)  # 16ch

    # ----- 패치 추론 -----
    pred = patch_inference(realA_img, realA_dis, netG, PATCH_SIZE, OVERLAP, SW_BATCH)
    pred_np = pred.squeeze().cpu().numpy()                # [D,H,W]
    pred_np = np.transpose(pred_np, (2, 1, 0))            # [z,y,x]
    # pred_np[pred_np > np.max(hptv_npy*100)] = np.max(hptv_npy*100)
    out_itk = sitk.GetImageFromArray(pred_np)
    out_itk.CopyInformation(itk_ref)
    save_path = os.path.join(SAVE_DIR, f"{pid}_dose_output.nii.gz")
    sitk.WriteImage(out_itk, save_path)
    print(f"   저장 → {save_path}")

print("\n전체 완료!")
