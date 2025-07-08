
from torch import tensor
import warnings
import numpy as np

from torch.utils.data import DataLoader
from torchsig.utils.random import Seedable

def metadata_padding_collate_fn(batch):

    default_y_value = 0

    batch_max_len = 0
    iqs = []
    y_tensor_obj = {}
    for data_pair in batch:
        if not isinstance(data_pair, tuple) or len(data_pair) != 2:
            raise ValueError(str(data_pair) + " is not a valid (x, y) pair; this collate function expects datasets to return tuples of (x, y)")
        if batch_max_len < len(data_pair[1]):
            batch_max_len = len(data_pair[1])
        for metadata_obj in data_pair[1]:
            for key in metadata_obj.keys():
                if not key in y_tensor_obj.keys():
                    y_tensor_obj[key] = []
        iqs += [data_pair[0]]

    if batch_max_len < 1:
        return tensor(np.array(iqs)), y_tensor_obj
    
    for key in y_tensor_obj.keys():
        y_tensor_obj[key] = [[]]*batch_max_len
        
    for data_pair in batch:
        for i in range(batch_max_len):
            if len(data_pair[1]) > i:
                #add the record from the metadata
                metadata_obj = data_pair[1][i]
                for key in y_tensor_obj.keys():
                    if key in metadata_obj.keys():
                        y_tensor_obj[key][i] += [metadata_obj[key]]
                    else:
                        y_tensor_obj[key][i] += [default_y_value]
            else:
                #add a record consisting entirely of default values  
                for key in y_tensor_obj.keys():
                    y_tensor_obj[key][i] += [default_y_value]
    
    final_tensor_obj = {}
    for key in y_tensor_obj:
        try:
            final_tensor_obj[key] = tensor(np.array(y_tensor_obj[key]))
        except:
            warnings.warn("Dropping key value: '"+key+"' because it contained invalid tensor values")

    return tensor(np.array(iqs)), final_tensor_obj

class WorkerSeedingDataLoader(DataLoader, Seedable):
    """
    A Custom DataLoader for torchsig that seeds workers differently on worker init based on a shared initial seed;
    """

    def __init__(self, dataset, collate_fn=metadata_padding_collate_fn, **kwargs):
        DataLoader.__init__(self, dataset, collate_fn=collate_fn, **kwargs)
        Seedable.__init__(self, **kwargs)
        if self.worker_init_fn:
            raise ValueError("No worker_init_fn should be given to WorkerSeedingDataLoader; it will set it's own worker_init_fn.")
        self.worker_init_fn = self.init_worker_seed

    def init_worker_seed(self, worker_id):
        from torch.utils.data import get_worker_info
        get_worker_info().dataset.seed(int(self.random_generator.random()*100 + 1) * (worker_id + 1))

    def __len__(self):
        return self.dataset.dataset_metadata.num_samples


