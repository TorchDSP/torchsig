from torchsig.transforms.target_transforms import DescToBBoxYoloSignalDict
from torchsig.transforms import Spectrogram, Normalize, SpectrogramImage
from torchsig.datasets.wideband_sig53 import WidebandSig53
from torchsig.transforms.transforms import Compose as CP
from torchsig.datasets import conf
import torchaudio

from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils.torch_utils import de_parallel
from ultralytics.utils import DEFAULT_CFG, colorstr

from copy import copy, deepcopy
import concurrent.futures
from tqdm import tqdm  
from pathlib import Path
import numpy as np
import os
import random
import torch
import cv2

from ultralytics.nn.tasks import DetectionModel

class TorchsigDataset(YOLODataset):
    def __init__(self, *args, mode='train', imgsz=640, hyp=DEFAULT_CFG, data=None, task="detect", **kwargs):
        """
        Initializes the TorchsigDataset, which inherits from YOLODataset. This custom dataset class is tailored 
        to handle spectrogram data and corresponding bounding box labels using the WidebandSig53 dataset.

        Args:
            mode (str): Indicates whether the dataset is for training ('train') or validation/testing ('val' or 'test').
            imgsz (int): Size of the input image (assumed to be square). Default is 640.
            hyp (dict): Hyperparameters for the YOLO model. Default is `DEFAULT_CFG`.
            data (dict): Dictionary containing paths to the data for different modes.
            task (str): Task type, e.g., "detect". Default is "detect".
            *args: Additional arguments for the parent YOLODataset class.
            **kwargs: Additional keyword arguments for the parent YOLODataset class.
        """
        self.mode = mode  # Store mode (train, val, test)
        self.root = data[self.mode]  # Get the data path for the current mode
        self.train = True if mode == 'train' else False  # Boolean indicating training mode

        # Define the transformations for spectrogram and normalization
        ts_transform = CP([
            # Spectrogram(nperseg=imgsz, noverlap=0, nfft=imgsz, mode='psd'),
            # Normalize(norm=np.inf, flatten=True),
            # SpectrogramImage(),
        ])

        # Define the transformation for converting description to bounding boxes
        target_transform = CP([
            DescToBBoxYoloSignalDict()
        ])

        # Initialize the WidebandSig53 dataset with the defined transforms
        self.wbsig53 = WidebandSig53(
            root=self.root,
            train=self.train,
            impaired=True,
            transform=ts_transform,
            target_transform=target_transform
        )
        self.spectrogram = torchaudio.transforms.Spectrogram(n_fft=imgsz, win_length=imgsz, hop_length=imgsz, normalized=True, center=False, onesided=False, power=True)


        # Initialize the parent YOLODataset class
        super().__init__(data=data, imgsz=imgsz, *args, **kwargs)

    def get_labels(self, batch_size=32, num_threads=32):
        """
        Constructs and returns a list of labels for the dataset using batching and multithreading.
    
        Args:
            batch_size (int): The size of each batch.
            num_threads (int): The number of threads to use for multithreading.
    
        Returns:
            labels (list): A list of dictionaries, each containing label information for an image.
        """
        cache_path = Path(self.root + '/' + self.mode + '.cache')
        if cache_path.exists():
            print(f'Loading label caches from {self.root}')
            x = load_dataset_cache_file(cache_path)
        else:
            x = {'labels': []}  # Initialize a dictionary to store labels
            num_samples = self.wbsig53.length
            
            # Create a list of (start_idx, end_idx) tuples for each batch
            batches = [(i, min(i + batch_size, num_samples)) for i in range(0, num_samples, batch_size)]
        
            # Use multithreading to process each batch
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:    
                with tqdm(total=len(batches), desc=f"Fetching labels for: {self.mode}") as pbar:
                    for batch_labels in executor.map(lambda b: process_batch(*b, self.wbsig53, self.root, self.mode, self.imgsz), batches):
                        x['labels'].extend(batch_labels)
                        pbar.update(1) 
            print(f'Caching labels to {cache_path}')
            save_dataset_cache_file(cache_path, x)
        
        labels = x['labels']
        return labels

    def get_img_files(self, img_path):
        """
        Retrieves the list of image file paths.

        Args:
            img_path (str): Path to the directory containing images.

        Returns:
            im_files (list): A list containing the current working directory as a placeholder.
        """
        im_files = [Path.cwd()]  # Placeholder for image files; replace with actual implementation
        return im_files

    def set_rectangle(self):
        """
        Sets the shape of bounding boxes for YOLO detections as rectangles, based on the aspect ratio 
        of the images. This is useful for rectangular batch processing in YOLO.
        """
        bi = np.floor(np.arange(self.ni) / self.batch_size).astype(int)  # Batch index
        nb = bi[-1] + 1  # Number of batches

        s = np.array([x.pop("shape") for x in self.labels])  # Extract image shapes (height, width)
        ar = s[:, 0] / s[:, 1]  # Calculate aspect ratio
        irect = ar.argsort()  # Sort indices by aspect ratio
        self.labels = [self.labels[i] for i in irect]  # Reorder labels by aspect ratio
        ar = ar[irect]  # Reorder aspect ratios

        # Initialize shapes for rectangular batches
        shapes = [[1, 1]] * nb
        for i in range(nb):
            ari = ar[bi == i]
            mini, maxi = ari.min(), ari.max()
            if maxi < 1:
                shapes[i] = [maxi, 1]
            elif mini > 1:
                shapes[i] = [1, 1 / mini]

        # Set batch shapes based on image sizes and strides
        self.batch_shapes = np.ceil(np.array(shapes) * self.imgsz / self.stride + self.pad).astype(int) * self.stride
        self.batch = bi  # Batch index of images

    def get_image_and_label(self, index):
        """
        Retrieves the image and corresponding label information from the dataset.

        Args:
            index (int): Index of the image/label pair to retrieve.

        Returns:
            label (dict): A dictionary containing the image, its original shape, resized shape, 
                          and other relevant information.
        """
        label = deepcopy(self.labels[index])  # Deep copy label to avoid modifications to the original
        data, _ = self.wbsig53[label['idx']] # Retrieve image and label using the dataset index

        
        label.pop("shape", None)  # Remove the shape key as it's used only for rectangle mode
        label["img"], label["ori_shape"], label["resized_shape"] = self.load_image(index, data)  # Load and resize image
        label["ratio_pad"] = (
            label["resized_shape"][0] / label["ori_shape"][0],
            label["resized_shape"][1] / label["ori_shape"][1],
        )  # Ratio of resizing for evaluation

        if self.rect:
            label["rect_shape"] = self.batch_shapes[self.batch[index]]  # Set rectangle shape if in rectangle mode
        return self.update_labels_info(label)  # Update and return label information

    def load_image(self, i, data, rect_mode=True):
        """
        Loads and resizes an image from the dataset. The image is resized based on the specified
        mode (rectangular or square).

        Args:
            i (int): Index of the image to load.
            im (ndarray): The image to be loaded.
            rect_mode (bool): Whether to resize the image while maintaining aspect ratio.

        Returns:
            im (ndarray): The resized image.
            (h0, w0) (tuple): The original height and width of the image.
            im.shape[:2] (tuple): The resized image shape.
        """
        spec = self.spectrogram(torch.from_numpy(data))
        spec = torch.fft.fftshift(spec, dim=0)
        spec = 10*np.log10(spec.numpy())
        img = np.zeros((spec.shape[0], spec.shape[1], 3), dtype=np.float32)
        img = cv2.normalize(spec, img, 0, 255, cv2.NORM_MINMAX)
        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        
        h0, w0 = img.shape[:2]  # Original height and width

        # Resize image if in rectangular mode
        if rect_mode:
            r = self.imgsz / max(h0, w0)  # Calculate resize ratio
            if r != 1:  # If resize is needed
                w, h = (min(int(w0 * r), self.imgsz), min(int(h0 * r), self.imgsz))
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        elif not (h0 == w0 == self.imgsz):  # Resize to square if not already square
            img = cv2.resize(img, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)

        # Buffer the image if using augmentations during training
        if self.augment:
            self.ims[i], self.im_hw0[i], self.im_hw[i] = img, (h0, w0), img.shape[:2]  # Cache image and shapes
            self.buffer.append(i)  # Add index to buffer
            if 1 < len(self.buffer) >= self.max_buffer_length:  # Limit buffer size
                j = self.buffer.pop(0)  # Remove oldest entry
                if self.cache != "ram":
                    self.ims[j], self.im_hw0[j], self.im_hw[j] = None, None, None  # Free memory
        return img, (h0, w0), img.shape[:2]  # Return image and shape information

def build_torchsig_dataset(cfg, img_path, batch, data, mode="train", rect=False, stride=32, multi_modal=False):
    """
    Builds and returns a TorchsigDataset for use in a YOLO detection model.

    Args:
        cfg (object): Configuration object containing hyperparameters and settings.
        img_path (str): Path to the image files or dataset root directory.
        batch (int): Batch size for training or validation.
        data (dict): Dictionary containing dataset paths and other relevant information.
        mode (str, optional): Mode of operation ('train' or 'val'). Defaults to 'train'.
        rect (bool, optional): If True, uses rectangular training or validation shapes. Defaults to False.
        stride (int, optional): Stride size for the model. Defaults to 32.
        multi_modal (bool, optional): If True, enables multi-modal data handling. Defaults to False.

    Returns:
        TorchsigDataset: An instance of the TorchsigDataset class.
    """
    dataset = TorchsigDataset
    return dataset(
        img_path=img_path,
        imgsz=cfg.imgsz,  # Image size (resolution) to use for the model
        batch_size=batch,  # Batch size for processing
        augment=mode == "train",  # Enable data augmentation if in training mode
        hyp=cfg,  # Hyperparameters for the dataset
        rect=cfg.rect or rect,  # Use rectangular image shapes if specified
        mode=mode,  # Operation mode: 'train' or 'val'
        cache=None,  # No caching by default
        single_cls=cfg.single_cls or False,  # Treat all classes as a single class if specified
        stride=int(stride),  # Stride size used by the model
        pad=0.0 if mode == "train" else 0.5,  # Padding applied to images (less padding during training)
        prefix=colorstr(f"{mode}: "),  # Prefix for logging messages
        task=cfg.task,  # Task type ('detect', etc.)
        classes=cfg.classes,  # Number of classes in the dataset
        data=data,  # Data dictionary containing dataset information
        fraction=cfg.fraction if mode == "train" else 1.0,  # Fraction of the dataset to use during training
    )

def process_batch(start_idx, end_idx, dataset, root, mode, imgsz):
    """
    Process a batch of labels from the dataset.

    Args:
        start_idx (int): Start index for the batch.
        end_idx (int): End index for the batch.
        wbsig53 (Dataset): The dataset object.
        root (str): Root directory for the image files.
        mode (str): Mode indicating the dataset type.
        imgsz (int): Image size.

    Returns:
        list: A list of dictionaries containing label information for the batch.
    """
    batch_labels = []

    for y in range(start_idx, end_idx):
        _, lbls = dataset[y]  # Get the label dictionary from the dataset
        lbl_arr = lbls['labels'].numpy().astype(np.float32)  # Convert labels to a numpy array
        lbl_arr = lbl_arr.reshape((lbl_arr.shape[0], 1))  # Reshape the label array
        im_name = root + '_' + mode + '_' + str(y)
        # Append the label information to the list
        batch_labels.append(
            {
                "im_file": im_name,
                "idx": y,
                # "img": img,
                "shape": (imgsz, imgsz),
                "cls": lbl_arr,  # n, 1
                "bboxes": lbls['boxes'].numpy(),  # n, 4
                "segments": [],
                "keypoints": None,
                "normalized": True,
                "bbox_format": "xywh",
            }
        )
    
    return batch_labels

def load_dataset_cache_file(path):
    """Load an Torchsig/Ultralytics *.cache dictionary from path."""
    import gc

    gc.disable()  # reduce pickle load time https://github.com/ultralytics/ultralytics/pull/1585
    cache = np.load(str(path), allow_pickle=True).item()  # load dict
    gc.enable()
    return cache


def save_dataset_cache_file(path, x):
    """Save an Torchsig dataset *.cache dictionary x to path."""
    np.save(str(path), x)  # save cache for next time
    path.with_suffix(".cache.npy").rename(path)  # remove .npy suffix
    print(f"New cache created: {path}")

class Yolo_Trainer(DetectionTrainer):
    def build_dataset(self, img_path, mode="train", batch=None):
        """
        Builds the YOLO Torchsig dataset using the specified parameters.

        Args:
            img_path (str): Path to the directory containing images.
            mode (str): Mode of operation ('train' or 'val'). Determines whether to apply augmentations.
            batch (int, optional): Batch size for training or validation. Defaults to None.

        Returns:
            TorchsigDataset: An instance of the TorchsigDataset class configured for the specified mode.
        """
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        # Grid stride size, determined by the model's stride, with a minimum of 32
        return build_torchsig_dataset(
            self.args,  # Configuration arguments
            img_path,  # Path to the dataset images
            batch,  # Batch size
            self.data,  # Data configuration
            mode=mode,  # Mode of operation: 'train' or 'val'
            rect=mode == "val",  # Use rectangular images during validation
            stride=gs  # Stride size for the model
        )

