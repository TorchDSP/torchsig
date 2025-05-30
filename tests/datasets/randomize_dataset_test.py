#!/usr/bin/env python3

# testing dataset with randomized parameters

from torchsig.datasets.dataset_metadata import DatasetMetadata
from torchsig.datasets.datasets import NewTorchSigDataset, StaticTorchSigDataset
from torchsig.utils.writer import DatasetCreator
from torchsig.signals.signal_lists import TorchSigSignalLists
from torchsig.transforms.target_transforms import (
    ClassName,
    Start,
    Stop,
    LowerFreq,
    UpperFreq,
    SNR,
    YOLOLabel
)
from torchsig.transforms.transforms import Spectrogram

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import os
import shutil

# number of samples to test generation
num_samples = 10
save_num_signals = 5

# signals to simulate
class_list = TorchSigSignalLists.all_signals

# distribution of classes
class_dist = np.ones(len(class_list))/len(class_list)

# FFT/spectrogram params
fft_size = np.random.randint(128,1024)
# num_iq_samples_dataset = fft_size*np.random.randint(128,1024)
num_iq_samples_dataset = fft_size**2

# testing to handle cases in which number of samples is not an integer multiple of FFT size
num_iq_samples_dataset += np.random.randint(0,fft_size) # test cases in which data length is not integer multiple of FFT size

# works for variable sample rates
sample_rate = 100e6

# minimum and maximum SNR for signals
snr_db_max = 50
snr_db_min = 0

# define impairment level
impairment_level = 2

# define maximum number of signals to generate
num_signals_min = 1
num_signals_max = 10

# probability for each sample to contain N signals where N is the index,
# for example, num_signals_dist = [0.15, 0.5, 0.35] is 25% probability to 
# generate 0 signals, 50% probability to generate 1 signal, 35% 
# probability to generate 2 signals
num_signals_dist = np.ones(num_signals_max - num_signals_min+1)/(num_signals_max-num_signals_min+1)

# define transforms
transforms = Spectrogram(fft_size=fft_size)
target_transform = [
    ClassName(),
    Start(),
    Stop(),
    LowerFreq(),
    UpperFreq(),
    SNR(),
    YOLOLabel()
]

# set up path to cache directory
root = Path.joinpath(Path(__file__).parent,'wideband_test')
image_path = f"{root}/images_impaired_{impairment_level}"

if os.path.exists(root):
    shutil.rmtree(f"{root}")
os.makedirs(root, exist_ok=True)
os.makedirs(image_path, exist_ok=True)

# build the wideband metadata
md = DatasetMetadata(
    num_iq_samples_dataset=num_iq_samples_dataset,
    sample_rate=sample_rate,
    fft_size=fft_size,
    num_samples=num_samples,
    num_signals_max=num_signals_max,
    num_signals_min=num_signals_min,
    num_signals_distribution=num_signals_dist,
    snr_db_max=snr_db_max,
    snr_db_min=snr_db_min,
    transforms=transforms,
    target_transforms=target_transform,
    impairment_level=impairment_level,
    class_list=class_list,
    class_distribution=class_dist,
)

# create the wideband object, derived from the metadata object
WB = NewTorchSigDataset(dataset_metadata=md)


# write dataset to disk
dc = DatasetCreator(
    dataset=WB,
    root=root,
    overwrite=True,
    batch_size=3,
    num_workers=4
)
dc.create()

# load dataset back in from disk
WBS = StaticTorchSigDataset(
    root=root,
    impairment_level=impairment_level,
)

# save as images
for i in tqdm(range(save_num_signals), desc = "Saving as Images"):
    data, targets = WBS[i] # (data, List[dict])

    fig = plt.figure(figsize=(18,12))
    ax = fig.add_subplot(1,1,1)
    xmin = 0
    xmax = 1
    ymin = -sample_rate / 2
    ymax = sample_rate / 2
    pos = ax.imshow(data,extent=[xmin,xmax,ymin,ymax],aspect='auto',cmap='Wistia',vmin=md.noise_power_db)

    fig.colorbar(pos, ax=ax)

    title = "labels: "

    for t in targets:
        classname, start, stop, lower, upper, snr, yololabel = t

        ax.plot([start,start],[lower,upper],'b',alpha=0.5)
        ax.plot([stop, stop],[lower,upper],'b',alpha=0.5)
        ax.plot([start,stop],[lower,lower],'b',alpha=0.5)
        ax.plot([start,stop],[upper,upper],'b',alpha=0.5)
        textDisplay = str(classname) + ', SNR = ' + str(snr) + ' dB'
        ax.text(start,lower,textDisplay, bbox=dict(facecolor='w', alpha=0.5, linewidth=0))
        ax.set_xlim([0,1])
        ax.set_ylim([-sample_rate/2,sample_rate/2])
        title = f"{title}{classname} "
    
    fig.suptitle(title, fontsize=16)
    plt.ylabel("Frequency (Hz)")
    plt.xlabel("Time")
    plt.savefig(f"{image_path}/{i}")
    plt.close()

    

