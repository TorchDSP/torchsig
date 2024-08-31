from torchsig.datasets.modulations import ModulationsDataset
from torchsig.transforms.target_transforms import DescToFamilyName
from torchsig.transforms.transforms import Compose as CP
from torchsig.utils.yolo_val import ClassificationValidator

from ultralytics.models.yolo.classify import ClassificationTrainer
from ultralytics.utils import DEFAULT_CFG
from ultralytics.data.augment import classify_augmentations, classify_transforms

from typing import Callable, Optional
from scipy.signal import ShortTimeFFT
from PIL import Image
import yaml
import numpy as np
import cv2


class TorchsigClassificationDataset:
    """
    A dataset class for handling signal data and transforming it into spectrogram images
    suitable for classification tasks using YOLO models.

    Attributes:
        config (dict): Configuration loaded from the dataset root.
        class_list (List[str]): List of class names.
        class_to_idx_dict (dict): Mapping of class names to indices.
        samples (ModulationsDataset): Dataset containing the signal samples.
        root (str): Root directory of the dataset.
        torch_transforms (Callable): Transformations applied to images.
        image_transform (Optional[Callable]): Optional custom transformation function for images.
    """

    def __init__(self, root, args, augment=False, image_transform=None):
        """
        Initializes the TorchsigClassificationDataset class.

        Args:
            root (str): Path to the dataset configuration file.
            args (Namespace): Arguments containing dataset parameters.
            augment (bool): Flag indicating whether to apply data augmentation.
            image_transform (Optional[Callable]): Optional custom image transformation function.
        """
        # Load the dataset configuration from the root file
        with open(root, 'r') as file:
            self.config = yaml.safe_load(file)

        # Create a list of class names
        self.class_list = [item[1] for item in self.config['names'].items()]

        # Determine whether to map descriptions to family names
        if self.config['family']:
            self.class_to_idx_dict = {v: k for k, v in self.config['family'].items()}
            target_transform = CP([DescToFamilyName()])
        else:
            self.class_to_idx_dict = {v: k for k, v in self.config['names'].items()}
            target_transform = None

        # Initialize the ModulationsDataset with provided configurations
        dataset = ModulationsDataset(
            classes=self.class_list,
            use_class_idx=False,
            level=self.config['level'],
            num_iq_samples=args.imgsz**2,
            num_samples=int(self.config['nc'] * 1000),
            include_snr=self.config['include_snr'],
            target_transform=target_transform
        )

        self.samples = dataset
        self.root = root

        # Set up data augmentation and image transformations
        scale = (1.0 - args.scale, 1.0)
        self.torch_transforms = (
            classify_augmentations(
                size=args.imgsz,
                scale=scale,
                hflip=args.fliplr,
                vflip=args.flipud,
                erasing=args.erasing,
                auto_augment=args.auto_augment,
                hsv_h=args.hsv_h,
                hsv_s=args.hsv_s,
                hsv_v=args.hsv_v,
            )
            if augment
            else classify_transforms(size=args.imgsz, crop_fraction=args.crop_fraction)
        )

        # Initialize the spectrogram image transformation
        if image_transform is None:
            window = np.blackman(args.imgsz)
            self.STF = ShortTimeFFT(
                win=window, hop=args.imgsz + 1, fs=args.imgsz**2, fft_mode='centered', scale_to='psd'
            )
            self.image_transform = self.spectrogram_image
        else:
            self.image_transform = image_transform

    def __getitem__(self, i):
        """
        Retrieves a sample from the dataset and applies the necessary transformations.

        Args:
            i (int): Index of the sample to retrieve.

        Returns:
            dict: A dictionary containing the transformed image and its corresponding class index.
        """
        data, label_name = self.samples[i]  # Retrieve data and label name for the given index
        j = self.class_to_idx_dict[label_name]  # Get class index from the label name
        im = self.image_transform(data)  # Apply the image transformation
        im = Image.fromarray(im)  # Convert the numpy array to a PIL Image
        sample = self.torch_transforms(im)  # Apply transformations to the image
        return {"img": sample, "cls": j}

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.samples)

    def spectrogram_image(self, signal):
        """
        Converts a given signal into a spectrogram image.

        Args:
            signal (np.ndarray): The input signal to be converted.

        Returns:
            np.ndarray: The resulting spectrogram image.
        """
        # Calculate the spectrogram from the signal
        spec = self.STF.spectrogram(signal)
        spec = 10 * np.log10(spec)  # Convert to dB scale

        # Initialize an image array and normalize the spectrogram data
        img = np.zeros((spec.shape[0], spec.shape[1], 3), dtype=np.float32)
        img = cv2.normalize(spec, img, 0, 255, cv2.NORM_MINMAX)
        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR)

        return img


class YoloClassifyTrainer(ClassificationTrainer):
    """
    Custom trainer class for signal classification tasks using the YOLO model.

    Attributes:
        image_transform (Optional[Callable]): Optional custom transformation function for images.
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None, image_transform: Optional[Callable] = None):
        """
        Initializes the YoloClassifyTrainer class.

        Args:
            cfg (dict): Default configuration for the YOLO model.
            overrides (Optional[dict]): Configuration overrides.
            _callbacks (Optional[list]): List of callback functions.
            image_transform (Optional[Callable]): Optional custom image transformation function.
        """
        if overrides is None:
            overrides = {}
        overrides["task"] = "classify"  # Set task to classify
        if overrides.get("imgsz") is None:
            overrides["imgsz"] = 224  # Set default image size
        self.image_transform = image_transform
        super().__init__(cfg, overrides, _callbacks)

    def build_dataset(self, img_path, mode="train", batch=None):
        """
        Creates a dataset for training or validation.

        Args:
            img_path (str): Path to the dataset.
            mode (str): Mode of operation, either 'train' or 'val'.
            batch (Optional[int]): Batch size for the dataset.

        Returns:
            TorchsigClassificationDataset: The constructed dataset.
        """
        print(f'args -> {img_path}')
        return TorchsigClassificationDataset(
            root=img_path, args=self.args, augment=mode == "train", image_transform=self.image_transform
        )

    def get_dataset(self):
        """
        Retrieves the dataset paths for training, validation, and testing.

        Returns:
            Tuple[str, Optional[str]]: Paths to the training and validation datasets.
        """
        with open(self.args.data, 'r') as file:
            config = yaml.safe_load(file)
        names = config['names']
        nc = config['nc']
        data = {"train": self.args.data, "val": self.args.data, "test": self.args.data, "nc": nc, "names": names}
        self.data = data
        return data["train"], data.get("val") or data.get("test")

    def get_validator(self):
        """
        Returns a validator for evaluating the classification model.

        Returns:
            ClassificationValidator: Validator for classification tasks.
        """
        self.loss_names = ["loss"]  # Define the loss name for the task
        return ClassificationValidator(self.test_loader, self.save_dir, _callbacks=self.callbacks)
