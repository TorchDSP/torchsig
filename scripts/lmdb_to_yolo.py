

"""
Pseudo code outline
-------------------

Assuming IQ data has already been generated, convert the IQ data to YOLO data.




"""

from torchsig.datasets.datamodules import WidebandDataModule
from torch.utils.data import DataLoader
from torchsig.utils.dataset import collate_fn
from torchsig.datasets.torchsig_narrowband import TorchSigNarrowband
from torchsig.datasets.torchsig_wideband import TorchSigWideband
from torchsig.datasets.signal_classes import torchsig_signals
from torchsig.transforms.target_transforms import DescToListTuple, ListTupleToYOLO
from torchsig.transforms.transforms import Spectrogram, SpectrogramImage, Normalize, Compose, Identity
import pytorch_lightning as pl
import numpy as np

from ultralytics import YOLO
import cv2
import yaml
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import torch

# GENERATE DATA AS SPECTROGRAM

root = './datasets/wideband'            # Location in which data to be converted is stored
fft_size = 512
num_classes = len(torchsig_signals.class_list)
impaired = True
num_workers = 4
batch_size = 1
qa = True

transform = Compose([
    Normalize(norm=np.inf, flatten=True),
    Spectrogram(nperseg=fft_size, noverlap=0, nfft=fft_size, detrend=None),
    Normalize(norm=np.inf, flatten=True),
    SpectrogramImage(),    
])

target_transform = Compose([
    DescToListTuple(),
    ListTupleToYOLO()
])

# Instantiate the TorchSigWideband Dataset
datamodule = WidebandDataModule(
    root=root,
    impaired=impaired,
    qa=qa,
    fft_size=fft_size,
    num_classes=num_classes,
    transform=transform,
    target_transform=target_transform,
    batch_size=batch_size,
    num_workers=num_workers
)
datamodule.prepare_data()
datamodule.setup("fit")

wideband_train = datamodule.train
wideband_val = datamodule.val


# Retrieve a sample and print out information
idx = np.random.randint(len(wideband_val))
data, label = wideband_val[idx]
print("Training Dataset length: {}".format(len(wideband_train)))
print("Validation Dataset length: {}".format(len(wideband_val)))
print("Data shape: {}\n\t".format(data.shape))
print(f"Label length: {len(label)}", end="\n\t")
print(*label, sep="\n\t")
print(f"Label: {type(label)} of {type(label[0])} \n")



# PREPARE DATA FOR YOLO

# method to output .png images and .txt label files in YOLO structure from wideband
def prepare_data(dataset: TorchSigWideband, output: str, train: bool, impaired: bool) -> None:
    output_root = os.path.join(output, "wideband_yolo")
    os.makedirs(output_root, exist_ok=True)
    impaired = "impaired" if impaired else "clean"
    train = "train" if train else "val"
    
    label_dir = os.path.join(output_root, impaired, "labels", train)
    image_dir = os.path.join(output_root, impaired, "images", train)
    os.makedirs(label_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)
    
    for i in tqdm(range(len(dataset))):
        image, labels = dataset[i]
        filename_base = str(i).zfill(10)
        label_filename = os.path.join(label_dir, filename_base) + ".txt"
        image_filename = os.path.join(image_dir, filename_base) + ".png"
        
        with open(label_filename, "w") as f:
            line = f""
            f.write("\n".join(f"{x[0]} {x[1]} {x[2]} {x[3]} {x[4]}" for x in labels))
            
        cv2.imwrite(image_filename, image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        
prepare_data(wideband_train, root, True, True)
prepare_data(wideband_val, root, False, True)

# create dataset yaml file
config_name = "05_yolo.yaml"
classes = {v: k for v, k in enumerate(torchsig_signals.class_list)}
classes[0] = 'signal'

wideband_yaml_dict = dict(
    path = "./wideband/wideband_yolo",
    train = "impaired/images/train",
    val = "impaired/images/val",
    nc = num_classes,
    names = classes
)

with open(config_name, 'w+') as f:
    yaml.dump(wideband_yaml_dict, f, default_flow_style=False)