# Variables
from torchsig.transforms.dataset_transforms import Spectrogram
from torchsig.transforms.target_transforms import YOLOLabel
from torchsig.signals.signal_lists import TorchSigSignalLists
from tqdm.notebook import tqdm
import numpy as np

root = "./datasets/impaired"
fft_size = 1024
num_iq_samples_dataset = fft_size ** 2
class_list = TorchSigSignalLists.all_signals
num_classes = len(class_list)
num_train = 350000 # size of train dataset
num_val = 35000 # size of validation dataset

# transform data into a spectrogram image
transforms = [Spectrogram(fft_size=fft_size)]
# YOLO labels are expected to be (class index, x center, y center, width, height)
# all normalized to zero, with (0,0) being upper left corner
target_transforms = [YOLOLabel()]

from torchsig.datasets.dataset_metadata import WidebandMetadata
from torchsig.datasets.datamodules import WidebandDataModule
from torchsig.datasets.wideband import NewWideband

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# create the NewWideband dataset
dataset_metadata = WidebandMetadata(
    num_iq_samples_dataset = num_iq_samples_dataset,
    fft_size = fft_size,
    impairment_level = 2,
    num_signals_max = 8,
    transforms=transforms,
    target_transforms=target_transforms,
)

wideband = NewWideband(
    dataset_metadata = dataset_metadata
)

import cv2
import shutil
import os

# writes the images to disk under images/
# writes the labels as a txt file under labels/
def prepare_dataset(dataset, train: bool, root: str, start_index: int, stop_index: int):
    os.makedirs(root, exist_ok = True)
    train_path = "train" if train else "val"
    label_dir = f"{root}/labels/{train_path}"
    image_dir = f"{root}/images/{train_path}"
    os.makedirs(label_dir, exist_ok = True)
    os.makedirs(image_dir, exist_ok = True)

    for i in tqdm(range(start_index, stop_index), desc=f"Writing YOLO {train_path.title()} Dataset"):
        image, labels = dataset[i]
        filename_base = str(i).zfill(10)
        label_filename = f"{label_dir}/{filename_base}.txt"
        image_filename = f"{image_dir}/{filename_base}.png"

        with open(label_filename, "w") as f:
            line = ""
            f.write("\n".join(f"{x[0]} {x[1]} {x[2]} {x[3]} {x[4]}" for x in labels))    
        img_new = np.zeros((image.shape[0], image.shape[1], 3),dtype=np.float32)    
        img_new = cv2.normalize(image, img_new, 0, 255, cv2.NORM_MINMAX)
        img_new = cv2.cvtColor(img_new.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        img_new = cv2.bitwise_not(img_new)
        cv2.imwrite(image_filename, img_new, [cv2.IMWRITE_PNG_COMPRESSION, 9])

if os.path.exists(root):
    shutil.rmtree(root)

prepare_dataset(wideband, train=True, root=root, start_index=0, stop_index = num_train)
prepare_dataset(wideband, train=False, root=root, start_index=num_train, stop_index = num_train + num_val)

# create dataset yaml file for ultralytics
import yaml
import torch

config_name = "wideband_detector_yolo.yaml"
classes = {v: k for v,k in enumerate(class_list)}

yolo_config = dict(
    path = "wideband_detector_example",
    train = "images/train",
    val = "images/val",
    nc = num_classes,
    names = classes
)

with open(config_name, 'w+') as file:
    yaml.dump(yolo_config, file, default_flow_style=False)

