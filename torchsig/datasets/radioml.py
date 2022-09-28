import h5py
import numpy as np
import pandas as pd
from typing import Tuple, Any, List, Optional, Callable

import torchsig.transforms as ST
from torchsig.utils.dataset import SignalDataset
from torchsig.utils.types import SignalData, SignalDescription


class RadioML2016(SignalDataset):
    """RadioML Dataset Example using RML2016.10a
    
    Args:
        root (:obj:`string`):
            Root directory where 'RML2016.10a_dict.pkl' exists. File can be downloaded from https://www.deepsig.ai/datasets

        classes (:obj:`list`):
            List of classes to retain. If None, full class list is returned.
            
        use_class_idx (:obj:`bool`):
            Return target as the class index rather than the class name. Default: False
            
        return_snr (:obj:`bool`):
            Return SNR as part of target. Default: False
            
        snr_threshold (:obj:`int`):
            Threshold to only return data entries with SNR values greater than or equal to threshold. If None, return all data entries.

        transform (callable, optional):
            A function/transform that takes in an IQ vector and returns a transformed version.
            
        target_transform (callable, optional):
            A function/transform that takes in the RFDescription object and transforms it to a custom target
        
    """
    def __init__(
        self,
        root: str,
        classes: Optional[List[str]] = None,
        use_class_idx: bool = False,
        include_snr: bool = False,
        snr_threshold: int = -2,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        super(RadioML2016, self).__init__()
        self.file_path = root + 'RML2016.10a_dict.pkl' 
        data = pd.read_pickle(self.file_path)
        snrs = []
        mods = []
        iq_data = []
        for k in data.keys():
            for idx in range(len(data[k])):
                mods.append(k[0])
                snrs.append(k[1])
                iq_data.append(np.asarray(data[k][idx][::2] + 1j*data[k][idx][1::2]).squeeze())
        data_dict = {'class_name':mods, 'snr':snrs, 'data':iq_data}
        self.data_table = pd.DataFrame(data_dict)
        
        if classes:
            classes = [x.upper() for x in classes]
            self.data_table = self.data_table[self.data_table.class_name.isin(classes)]
        if snr_threshold:
            self.data_table = self.data_table[self.data_table.snr>=snr_threshold]

        classes = list(self.data_table.class_name.unique())
        self.class_dict = dict(zip(classes, range(len(classes))))
        
        # Set the target transform based on input options if none provided
        if not target_transform:
            if use_class_idx:
                if include_snr:
                    self.target_transform = ST.DescToClassIndexSNR(class_list=classes)
                else:
                    self.target_transform = ST.DescToClassIndex(class_list=classes)
            else:
                if include_snr:
                    self.target_transform = ST.DescToClassNameSNR()
                else:
                    self.target_transform = ST.DescToClassName()
        self.transform = transform

    def __getitem__(self, item: int):
        signal_description = SignalDescription(
            sample_rate=0,
            samples_per_symbol=8,
            class_name=self.data_table["class_name"].iloc[item],
            snr=self.data_table["snr"].iloc[item],
        )        
        signal_data = SignalData(
            data=None,
            item_type=np.dtype(np.float64),
            data_type=np.dtype(np.complex128),
            signal_description=signal_description
        )
        signal_data.iq_data = self.data_table["data"].iloc[item] 

        if self.transform:
            signal_data = self.transform(signal_data)

        if self.target_transform:
            target = self.target_transform(signal_data.signal_description)
        else:
            target = signal_description

        return signal_data.iq_data, target        

    def __len__(self) -> int:
        return self.data_table.shape[0]
    

class RadioML2018(SignalDataset):
    """RadioML Dataset Example using RML2018.01
    
    Args:
        root (:obj:`string`):
            Root directory where 'GOLD_XYZ_OSC.0001_1024.hdf5' exists. File can be downloaded from https://www.deepsig.ai/datasets

        use_class_idx (:obj:`bool`):
            Return target as the class index rather than the class name. Default: False
            
        return_snr (:obj:`bool`):
            Return SNR as part of target. Default: False
            
        transform (callable, optional):
            A function/transform that takes in an IQ vector and returns a transformed version.
            
        target_transform (callable, optional):
            A function/transform that takes in the RFDescription object and transforms it to a custom target
        
    """
    def __init__(
        self,
        root: str,
        use_class_idx: bool = False,
        include_snr: bool = False,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        super(RadioML2018, self).__init__()
        path = root + 'GOLD_XYZ_OSC.0001_1024.hdf5'

        # Open the dataset
        hdf5_file = h5py.File(path,  'r')

        # Read the HDF5 groups
        self.data = hdf5_file['X']
        self.modulation_onehot = hdf5_file['Y']
        self.snr = hdf5_file['Z']
        
        # Class list corrected from `classes.txt` file
        self.class_list = [
            'OOK','4ASK','8ASK','BPSK','QPSK','8PSK','16PSK','32PSK','16APSK',
            '32APSK','64APSK','128APSK','16QAM','32QAM','64QAM','128QAM',
            '256QAM','AM-SSB-WC','AM-SSB-SC','AM-DSB-WC','AM-DSB-SC','FM',
            'GMSK','OQPSK',
        ]

        # Set the target transform based on input options if none provided
        if not target_transform:
            if use_class_idx:
                if include_snr:
                    self.target_transform = ST.DescToClassIndexSNR(class_list=self.class_list)
                else:
                    self.target_transform = ST.DescToClassIndex(class_list=self.class_list)
            else:
                if include_snr:
                    self.target_transform = ST.DescToClassNameSNR()
                else:
                    self.target_transform = ST.DescToClassName()
        self.transform = transform

    def __getitem__(self, item: int):
        signal_description = SignalDescription(
            sample_rate=0,
            samples_per_symbol=8,
            class_name=self.class_list[np.argmax(self.modulation_onehot[item])],
            snr=self.snr[item][0],
        )        
        signal_data = SignalData(
            data=None,
            item_type=np.dtype(np.float64),
            data_type=np.dtype(np.complex128),
            signal_description=signal_description
        )
        signal_data.iq_data = self.data[item][:,0] + 1j*self.data[item][:,1]

        if self.transform:
            signal_data = self.transform(signal_data)

        if self.target_transform:
            target = self.target_transform(signal_data.signal_description)
        else:
            target = signal_description

        return signal_data.iq_data, target        

    def __len__(self) -> int:
        return self.data.shape[0]
