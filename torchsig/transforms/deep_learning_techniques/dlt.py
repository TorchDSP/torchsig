import numpy as np
from copy import deepcopy
from typing import Tuple, List, Any, Union, Optional

from torchsig.utils import SignalDescription, SignalData, SignalDataset
from torchsig.transforms.transforms import SignalTransform
from torchsig.transforms.wireless_channel import TargetSNR
from torchsig.transforms.functional import to_distribution, uniform_continuous_distribution, uniform_discrete_distribution
from torchsig.transforms.functional import NumericParameter, FloatParameter
from torchsig.transforms.deep_learning_techniques import dlt_functional


class DatasetBasebandMixUp(SignalTransform):
    """Signal Transform that inputs a dataset to randomly sample from and insert 
    into the main dataset's examples, using the TargetSNR transform and the 
    additional `alpha` input to set the difference in SNRs between the two 
    examples with the following relationship:
    
       mixup_sample_snr = main_sample_snr + alpha
       
    Note that `alpha` is used as an additive value because the SNR values are
    expressed in log scale. Typical usage will be with with alpha values less
    than zero.
    
    This transform is loosely based on 
    `"mixup: Beyond Empirical Risk Minimization" <https://arxiv.org/pdf/1710.09412.pdf>`_.

    
    Args:
        dataset :obj:`SignalDataset`:
            A SignalDataset of complex-valued examples to be used as a source for
            the synthetic insertion/mixup
            
        alpha (:py:class:`~Callable`, :obj:`int`, :obj:`float`, :obj:`list`, :obj:`tuple`):
            alpha sets the difference in power level between the main dataset
            example and the inserted example
            * If Callable, produces a sample by calling target_snr()
            * If int or float, target_snr is fixed at the value provided
            * If list, target_snr is any element in the list
            * If tuple, target_snr is in range of (tuple[0], tuple[1])
    
    Example:
        >>> import torchsig.transforms as ST
        >>> from torchsig.datasets import ModulationsDataset
        >>> # Add signals from the `ModulationsDataset`
        >>> target_transform = SignalDescriptionPassThroughTransform()
        >>> dataset = ModulationsDataset(
                            use_class_idx=True,
                            level=0,
                            num_iq_samples=4096,
                            num_samples=5300,
                            target_transform=target_transform,
                            )
        >>> transform = ST.DatasetBasebandMixUp(dataset=dataset,alpha=(-5,-3))
    
    """
    def __init__(
        self,
        dataset: SignalDataset = None,
        alpha: NumericParameter = uniform_continuous_distribution(-5, -3),
    ):
        super(DatasetBasebandMixUp, self).__init__()
        self.alpha = to_distribution(alpha, self.random_generator)
        self.dataset = dataset
        self.dataset_num_samples = len(dataset)
        
    def __call__(self, data: Any) -> Any:
        alpha = self.alpha()
        if isinstance(data, SignalData):
            # Input checks
            if len(data.signal_description) > 1:
                raise ValueError(
                    "Expected single `SignalDescription` for input `SignalData` but {} detected."
                    .format(len(data.signal_description))
                )    

            # Calculate target SNR of signal to be inserted
            target_snr_db = data.signal_description[0].snr + alpha

            # Randomly sample from provided dataset
            idx = np.random.randint(self.dataset_num_samples)
            insert_data, insert_signal_description = self.dataset[idx]
            if insert_data.shape[0] != data.iq_data.shape[0]:
                raise ValueError(
                    "Input dataset's `num_iq_samples` does not match main dataset.\n\t\
                    Found {}, but expected {} samples"
                    .format(insert_data.shape[0],data.shape[0])
                )
            insert_signal_data = SignalData(
                data=insert_data,
                item_type=np.dtype(np.float64),
                data_type=np.dtype(np.complex128),
                signal_description=insert_signal_description
            )
            
            # Set insert data's SNR
            target_snr_transform = TargetSNR(target_snr_db)
            insert_signal_data = target_snr_transform(insert_signal_data)
            
            # Create new SignalData object for transformed data
            new_data = SignalData(
                data=None,
                item_type=np.dtype(np.float64),
                data_type=np.dtype(np.complex128),
                signal_description=[],
            )
            new_data.iq_data = data.iq_data + insert_signal_data.iq_data
            
            # Update SignalDescription
            new_signal_description = []
            new_signal_description.append(data.signal_description[0])
            new_signal_description.append(insert_signal_data.signal_description[0])
            new_data.signal_description = new_signal_description

            return new_data
        else:
            raise ValueError(
                "Expected input type `SignalData`. Received {}. \n\t\
                The `SignalDatasetBasebandMixUp` transform depends on metadata from a `SignalData` object."
                .format(type(data))
            )
            
            
class DatasetBasebandCutMix(SignalTransform):
    """Signal Transform that inputs a dataset to randomly sample from and insert 
    into the main dataset's examples, using the TargetSNR transform to match
    the main dataset's examples' SNR and an additional `alpha` input to set the
    relative quantity in time to occupy, where
    
       cutmix_num_iq_samples = total_num_iq_samples * alpha
       
    With this transform, the inserted signal replaces the IQ samples of the 
    original signal rather than adding to them as the `DatasetBasebandMixUp`
    transform does above.
    
    This transform is loosely based on 
    `"CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features" <https://arxiv.org/pdf/1905.04899.pdf>`_.
    
    Args:
        dataset :obj:`SignalDataset`:
            An SignalDataset of complex-valued examples to be used as a source for
            the synthetic insertion/mixup
            
        alpha (:py:class:`~Callable`, :obj:`int`, :obj:`float`, :obj:`list`, :obj:`tuple`):
            alpha sets the difference in power level between the main dataset
            example and the inserted example
            * If Callable, produces a sample by calling target_snr()
            * If int or float, target_snr is fixed at the value provided
            * If list, target_snr is any element in the list
            * If tuple, target_snr is in range of (tuple[0], tuple[1])
    
    Example:
        >>> import torchsig.transforms as ST
        >>> from torchsig.datasets import ModulationsDataset
        >>> # Add signals from the `ModulationsDataset`
        >>> target_transform = SignalDescriptionPassThroughTransform()
        >>> dataset = ModulationsDataset(
                            use_class_idx=True,
                            level=0,
                            num_iq_samples=4096,
                            num_samples=5300,
                            target_transform=target_transform,
                            )
        >>> transform = ST.DatasetBasebandCutMix(dataset=dataset,alpha=(0.2,0.5))
    
    """
    def __init__(
        self,
        dataset: SignalDataset = None,
        alpha: NumericParameter = uniform_continuous_distribution(0.2, 0.5),
    ):
        super(DatasetBasebandCutMix, self).__init__()
        self.alpha = to_distribution(alpha, self.random_generator)
        self.dataset = dataset
        self.dataset_num_samples = len(dataset)
        
    def __call__(self, data: Any) -> Any:
        alpha = self.alpha()
        if isinstance(data, SignalData):
            # Input checks
            if len(data.signal_description) > 1:
                raise ValueError(
                    "Expected single `SignalDescription` for input `SignalData` but {} detected."
                    .format(len(data.signal_description))
                )    

            # Randomly sample from provided dataset
            idx = np.random.randint(self.dataset_num_samples)
            insert_data, insert_signal_description = self.dataset[idx]
            num_iq_samples = data.iq_data.shape[0]
            if insert_data.shape[0] != num_iq_samples:
                raise ValueError(
                    "Input dataset's `num_iq_samples` does not match main dataset.\n\t\
                    Found {}, but expected {} samples"
                    .format(insert_data.shape[0],data.shape[0])
                )
            insert_signal_data = SignalData(
                data=insert_data,
                item_type=np.dtype(np.float64),
                data_type=np.dtype(np.complex128),
                signal_description=insert_signal_description
            )
            
            # Set insert data's SNR
            target_snr_transform = TargetSNR(data.signal_description[0].snr)
            insert_signal_data = target_snr_transform(insert_signal_data)
            
            # Mask both data examples based on alpha and a random start value
            insert_num_iq_samples = int(alpha * num_iq_samples)
            insert_start = np.random.randint(num_iq_samples - insert_num_iq_samples)
            insert_stop = insert_start+insert_num_iq_samples
            data.iq_data[insert_start:insert_stop] = 0
            insert_signal_data.iq_data[:insert_start] = 0
            insert_signal_data.iq_data[insert_stop:] = 0
            
            # Create new SignalData object for transformed data
            new_data = SignalData(
                data=None,
                item_type=np.dtype(np.float64),
                data_type=np.dtype(np.complex128),
                signal_description=[],
            )
            new_data.iq_data = data.iq_data + insert_signal_data.iq_data
            
            # Update SignalDescription
            new_signal_description = []
            if insert_start != 0 and insert_stop != num_iq_samples:
                # Data description becomes two SignalDescriptions
                new_signal_desc = deepcopy(data.signal_description[0])
                new_signal_desc.start = 0.0
                new_signal_desc.stop = insert_start / num_iq_samples
                new_signal_desc.duration = new_signal_desc.stop - new_signal_desc.start
                new_signal_description.append(new_signal_desc)
                new_signal_desc = deepcopy(data.signal_description[0])
                new_signal_desc.start = insert_stop / num_iq_samples
                new_signal_desc.stop = 1.0
                new_signal_desc.duration = new_signal_desc.stop - new_signal_desc.start
                new_signal_description.append(new_signal_desc)
            elif insert_start == 0:
                # Data description remains one SignalDescription up to end
                new_signal_desc = deepcopy(data.signal_description[0])
                new_signal_desc.start = insert_stop / num_iq_samples
                new_signal_desc.stop = 1.0
                new_signal_desc.duration = new_signal_desc.stop - new_signal_desc.start
                new_signal_description.append(new_signal_desc)
            else:
                # Data description remains one SignalDescription at beginning
                new_signal_desc = deepcopy(data.signal_description[0])
                new_signal_desc.start = 0.0
                new_signal_desc.stop = insert_start / num_iq_samples
                new_signal_desc.duration = new_signal_desc.stop - new_signal_desc.start
                new_signal_description.append(new_signal_desc)
            # Repeat for insert's SignalDescription
            new_signal_desc = deepcopy(insert_signal_data.signal_description[0])
            new_signal_desc.start = insert_start / num_iq_samples
            new_signal_desc.stop = insert_stop / num_iq_samples
            new_signal_desc.duration = new_signal_desc.stop - new_signal_desc.start
            new_signal_description.append(new_signal_desc)
            
            # Set output data's SignalDescription to above list
            new_data.signal_description = new_signal_description

            return new_data
        else:
            raise ValueError(
                "Expected input type `SignalData`. Received {}. \n\t\
                The `SignalDatasetBasebandCutMix` transform depends on metadata from a `SignalData` object."
                .format(type(data))
            )
            
            
class CutOut(SignalTransform):
    """A transform that applies the CutOut transform in the time domain. The
    `cut_dur` input specifies how long the cut region should be, and the 
    `cut_type` input specifies what the cut region should be filled in with.
    Options for the cut type include: zeros, ones, low_noise, avg_noise, and
    high_noise. Zeros fills in the region with zeros; ones fills in the region
    with 1+1j samples; low_noise fills in the region with noise with -100dB
    power; avg_noise adds noise at power average of input data, effectively
    slicing/removing existing signals in the most RF realistic way of the 
    options; and high_noise adds noise with 40dB power. If a list of multiple 
    options are passed in, they are randomly sampled from.
    
    This transform is loosely based on 
    `"Improved Regularization of Convolutional Neural Networks with Cutout" <https://arxiv.org/pdf/1708.04552v2.pdf>`_.
    
    Args:
         cut_dur (:py:class:`~Callable`, :obj:`int`, :obj:`float`, :obj:`list`, :obj:`tuple`):
            cut_dur sets the duration of the region to cut out
            * If Callable, produces a sample by calling cut_dur()
            * If int or float, cut_dur is fixed at the value provided
            * If list, cut_dur is any element in the list
            * If tuple, cut_dur is in range of (tuple[0], tuple[1])
            
        cut_type (:py:class:`~Callable`, :obj:`list`, :obj:`str`):
            cut_type sets the type of data to fill in the cut region with from
            the options: `zeros`, `ones`, `low_noise`, `avg_noise`, and
            `high_noise`
            * If Callable, produces a sample by calling cut_type()
            * If list, cut_type is any element in the list
            * If str, cut_type is fixed at the method provided
    
    """
    def __init__(
        self,
        cut_dur: NumericParameter = uniform_continuous_distribution(0.01,0.2),
        cut_type: Union[List, str] = uniform_discrete_distribution(["zeros", "ones", "low_noise", "avg_noise", "high_noise"]),
    ):
        super(CutOut, self).__init__()
        self.cut_dur = to_distribution(cut_dur, self.random_generator)
        self.cut_type = to_distribution(cut_type, self.random_generator)

    def __call__(self, data: Any) -> Any:
        cut_dur = self.cut_dur()
        cut_start = np.random.uniform(0.0,1.0-cut_dur)
        cut_type = self.cut_type()
        
        if isinstance(data, SignalData):
            # Create new SignalData object for transformed data
            new_data = SignalData(
                data=None,
                item_type=np.dtype(np.float64),
                data_type=np.dtype(np.complex128),
                signal_description=[],
            )
            
            # Update SignalDescription
            new_signal_description = []
            signal_description = [data.signal_description] if isinstance(data.signal_description, SignalDescription) else data.signal_description
            for signal_desc in signal_description:
                new_signal_desc = deepcopy(signal_desc)
                
                #Update labels
                if new_signal_desc.start > cut_start and new_signal_desc.start < cut_start + cut_dur:
                    # Label starts within cut region
                    if new_signal_desc.stop > cut_start and new_signal_desc.stop < cut_start + cut_dur:
                        # Label also stops within cut region --> Remove label
                        continue
                    else:
                        # Push label start to end of cut region
                        new_signal_desc.start = cut_start + cut_dur
                elif new_signal_desc.stop > cut_start and new_signal_desc.stop < cut_start + cut_dur:
                    # Label stops within cut region but does not start in region --> Push stop to begining of cut region
                    new_signal_desc.stop = cut_start
                elif new_signal_desc.start < cut_start and new_signal_desc.stop > cut_start + cut_dur:
                    # Label traverse cut region --> Split into two labels
                    new_signal_desc_split = deepcopy(signal_desc)
                    # Update first label region's stop
                    new_signal_desc.stop = cut_start
                    # Update second label region's start & append to description collection
                    new_signal_desc_split.start = cut_start + cut_dur
                    new_signal_description.append(new_signal_desc_split)
                
                new_signal_description.append(new_signal_desc)
                
            new_data.signal_description = new_signal_description
            
            # Perform data augmentation
            new_data.iq_data = dlt_functional.cut_out(data.iq_data, cut_start, cut_dur, cut_type)
                
        else:
            new_data = dlt_functional.cut_out(data, cut_start, cut_dur, cut_type)
        return new_data
    

class PatchShuffle(SignalTransform):
    """Randomly shuffle multiple local regions of samples.
    
    Transform is loosely based on 
    `"PatchShuffle Regularization" <https://arxiv.org/pdf/1707.07103.pdf>`_.

    Args:
         patch_size (:py:class:`~Callable`, :obj:`int`, :obj:`float`, :obj:`list`, :obj:`tuple`):
            patch_size sets the size of each patch to shuffle
            * If Callable, produces a sample by calling patch_size()
            * If int or float, patch_size is fixed at the value provided
            * If list, patch_size is any element in the list
            * If tuple, patch_size is in range of (tuple[0], tuple[1])
            
        shuffle_ratio (:py:class:`~Callable`, :obj:`int`, :obj:`float`, :obj:`list`, :obj:`tuple`):
            shuffle_ratio sets the ratio of the patches to shuffle
            * If Callable, produces a sample by calling shuffle_ratio()
            * If int or float, shuffle_ratio is fixed at the value provided
            * If list, shuffle_ratio is any element in the list
            * If tuple, shuffle_ratio is in range of (tuple[0], tuple[1])
    
    """
    def __init__(
        self,
        patch_size: NumericParameter = uniform_continuous_distribution(3,10),
        shuffle_ratio: FloatParameter = uniform_continuous_distribution(0.01,0.05),
    ):
        super(PatchShuffle, self).__init__()
        self.patch_size = to_distribution(patch_size, self.random_generator)
        self.shuffle_ratio = to_distribution(shuffle_ratio, self.random_generator)

    def __call__(self, data: Any) -> Any:
        patch_size = int(self.patch_size())
        shuffle_ratio = self.shuffle_ratio()
        
        if isinstance(data, SignalData):
            # Create new SignalData object for transformed data
            new_data = SignalData(
                data=None,
                item_type=np.dtype(np.float64),
                data_type=np.dtype(np.complex128),
                signal_description=data.signal_description,
            )
            
            # Perform data augmentation
            new_data.iq_data = dlt_functional.patch_shuffle(data.iq_data, patch_size, shuffle_ratio)
                
        else:
            new_data = dlt_functional.patch_shuffle(data, patch_size, shuffle_ratio)
        return new_data
    
