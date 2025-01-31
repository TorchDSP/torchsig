
# TorchSig
from torchsig.datasets.torchsig_wideband import TorchSigWideband
import torchsig.transforms.transforms as ST
import torchsig.transforms.target_transforms as TT

# Third Party
import numpy as np
from scipy import signal
from glob import glob
from tqdm import tqdm
import torch
import torchaudio
import pytorch_lightning as pl
import pandas as pd
import matplotlib.pyplot as plt
import cv2

# Built-In
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"



root = '.'
train = True
impaired = True
fft_size = 1024
num_classes = 53

transform = ST.Compose([
    ST.Normalize(norm=np.inf, flatten=True),
])

target_transform = ST.Compose([
    TT.DescToListTuple(),
])

wideband_dataset = TorchSigWideband(
    root=root,
    train=train,
    impaired=impaired,
    transform=transform,
    target_transform=target_transform,
)

output_dir_root = '.'
lbl_dir = output_dir_root+'/datasets/impaired/labels/train'
img_dir = output_dir_root+'/datasets/impaired/images/train'
os.makedirs(lbl_dir, exist_ok=True)
os.makedirs(img_dir, exist_ok=True)

modulation_list = [
    "ook",
    "bpsk",
    "4pam",
    "4ask",
    "qpsk",
    "8pam",
    "8ask",
    "8psk",
    "16qam",
    "16pam",
    "16ask",
    "16psk",
    "32qam",
    "32qam_cross",
    "32pam",
    "32ask",
    "32psk",
    "64qam",
    "64pam",
    "64ask",
    "64psk",
    "128qam_cross",
    "256qam",
    "512qam_cross",
    "1024qam",
    "2fsk",
    "2gfsk",
    "2msk",
    "2gmsk",
    "4fsk",
    "4gfsk",
    "4msk",
    "4gmsk",
    "8fsk",
    "8gfsk",
    "8msk",
    "8gmsk",
    "16fsk",
    "16gfsk",
    "16msk",
    "16gmsk",
    "ofdm-64",
    "ofdm-72",
    "ofdm-128",
    "ofdm-180",
    "ofdm-256",
    "ofdm-300",
    "ofdm-512",
    "ofdm-600",
    "ofdm-900",
    "ofdm-1024",
    "ofdm-1200",
    "ofdm-2048",
]   


for i in range(len(wideband_dataset)):
    data, annotation = wideband_dataset[i]
    X = str(i).zfill(10)
    lbl_name = lbl_dir + '/' + X + '.txt'
    img_name = img_dir + '/' + X + '.png'
    valid = True
    txt_box = []
    for lbl_obj_idx in range(len(annotation)):
        sig_class = annotation[lbl_obj_idx][0]
        start = annotation[lbl_obj_idx][1]
        stop = annotation[lbl_obj_idx][2]
        center_freq = annotation[lbl_obj_idx][3]
        bandwidth = annotation[lbl_obj_idx][4]
        lower_freq = center_freq - (bandwidth/2.0)
        upper_freq = center_freq + (bandwidth/2.0)
        center_time = (stop-start)/2.0
        duration = stop-start
        if upper_freq+0.5  <= 0 or lower_freq+0.5 >= 1:
            valid = False
            print(valid, 'frequency out of bounds')
        if duration <= 0:
            print(duration,'duration')
            valid = False
            print(valid, 'duration out of bounds')
        if valid == True:
            txt_box.append(str(modulation_list.index(sig_class)) \
                           +' '+str(start+0.5*duration)+' '+ \
                           str(np.clip(lower_freq + 0.5 + 0.5*bandwidth,0,1)) \
                           +' '+str(duration)+' '+str(bandwidth)+'\n')
    spectrogram = torchaudio.transforms.Spectrogram(
    	          n_fft=fft_size,
	              win_length=fft_size,
    	          hop_length=fft_size,
	              window_fn=torch.blackman_window,
	              normalized=False,
	              center=False,
	              onesided=False,
	              power=2,
	              )
    norm = lambda x: torch.linalg.norm(
	              x,
	              ord=float("inf"),
	              keepdim=True,
	              )      
    x = spectrogram(torch.from_numpy(data))
    x = x * (1 / norm(x.flatten()))
    x = torch.fft.fftshift(x,dim=0)
    x = 10*torch.log10(x+1e-12)
            
    with open(lbl_name, 'a') as lbl_file:
        for line in txt_box:
            lbl_file.write(line)


    img_new = np.zeros((fft_size, fft_size, 3),dtype=np.float32)
    img_new = cv2.normalize(x.numpy(), img_new, 0, 255, cv2.NORM_MINMAX)
    img_new = img_new.astype(np.uint8)
    img_new = cv2.bitwise_not(img_new)
    cv2.imwrite(img_name, img_new, [cv2.IMWRITE_PNG_COMPRESSION, 9])
