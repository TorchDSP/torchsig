import numpy as np
import torch
from torch.utils.data import Dataset

from torchsig.image_datasets.transforms.impairments import normalize_image

"""
Defines a concatinated Dataset of all the given datasets joined as one dataset object.
Inputs:
    component_datasets: a list of Dataset objects which contain instances of each component, represented as (image_component: ndarray(c,height,width), class_id: int)
    balance: whether or not to balance the dataset by selecting samples from component datasets at random
"""
class CombineDataset(Dataset):
    def __init__(self, component_datasets, balance = True, transforms = []):
        self.component_datasets = component_datasets
        self.balance = balance
        self.transforms = transforms
    def __len__(self):
        if self.balance:
            return max([len(ds) for ds in self.component_datasets])*len(self.component_datasets) #assume the length wed get if we upsampled other datasets to max the largest component dataset
        else:
            return np.sum([len(ds) for ds in self.component_datasets])
    def __getitem__(self, idx):
        x, y = None, None
        if self.balance: # ignores given index; will always be random with replacement
            ds = self.component_datasets[np.random.randint(0, len(self.component_datasets))]
            x, y = ds[np.random.randint(0, len(ds))]
        else:
            ds_ind = 0
            ds = self.component_datasets[ds_ind]
            ds_idx = idx
            while(ds_idx >= len(ds)):
                ds_idx -= len(ds)
                ds_ind += 1
                ds = self.component_datasets[ds_ind]
            x, y = ds[ds_idx]
        if self.transforms:
            if type(self.transforms) == list:
                for transform in self.transforms:
                    x = transform(x)
            else:
                x = self.transforms(x)
        return x, y
    def next(self):
        return self[np.random.randint(0,len(self))]


"""
Defines a component of a composite dataset; this will contain any information the composites should use to place instances of this component in the composites, such as how many instances should be place
Inputs:
    component_dataset: a Dataset object which contains instances of this component, represented as (image_component: ndarray(c,height,width), class_id: int)
    min_to_add: the fewest instances of this component type to be placed in each composite
    max_to_add: the most instances of this type to be placed in each composite; the number of instances will be selected unifomly from min_to_add to max_to_add
"""
class YOLOImageCompositeDatasetComponent(Dataset):
    def __init__(self, component_dataset, min_to_add=0, max_to_add=1):
        self.component_dataset = component_dataset
        self.min_to_add = min_to_add
        self.max_to_add = max_to_add
    def __len__(self):
        return len(self.component_dataset)
    def __getitem__(self, idx):
        return self.component_dataset[idx]
    def next(self):
        return self.component_dataset[np.random.randint(0,len(self.component_dataset))]
    def get_components_to_add(self):
        num_to_add = np.random.randint(self.min_to_add, self.max_to_add + 1)
        to_add = []
        for i in range(num_to_add):
            to_add += [self.next()]
        return to_add

"""
A Dataset class generating synthetic composite images in yolo format from other image datasets
Inputs:
    composite_scale: a tuple of the form (height, width, num_channels) specifying the scale of the image compisites to be generated; (if a 2d tuple is passed in, it will work in greyscale)
    transforms: either a single function or list of functions from images to images to be applied to each SOI; used for adding noise and impairments to data; defaults to None
    
    <NOTE>: The dataset will not have any components to add to the composite at initialization; these must be added by calling my_instance.add_component(image_dataset_to_add)
    All components should be torch datasets which output an image in the form of an ndarray and an integer class id label as: (image_height, image_width, ?image_depth), class_id
"""
class YOLOImageCompositeDataset(Dataset):
    def __init__(self, composite_scale, transforms=None, components = [], dataset_size = 10):
        self.composite_scale = composite_scale
        self.transforms = transforms
        self.components = []#components # list of YOLOImageCompositeDatasetComponent objects
        self.dataset_size = dataset_size
        
    def __len__(self):
        return self.dataset_size # placeholder value; this will generate new images, so there is in practice no fixed length of the dataset

    def add_component(self, component_dataset, min_to_add=0, max_to_add=1):
        self.components += [YOLOImageCompositeDatasetComponent(component_dataset, min_to_add=min_to_add, max_to_add=max_to_add)]
        
    def get_components_to_add(self):
        to_add = []
        for component in self.components:
            to_add += component.get_components_to_add()
        return to_add

    def add_component_to_image_and_labels(self,image, labels, component):
        component_image, component_label = component
        img_w = image.shape[-1]
        img_h = image.shape[-2]
        c_w = component_image.shape[-1]
        c_h = component_image.shape[-2]
        max_x = max(img_w - c_w, 0)
        max_y = max(img_h - c_h, 0)
        new_x = np.random.randint(0, max_x + 1)
        new_y = np.random.randint(0, max_y + 1)
        x_end = min(img_w, c_w + new_x)
        y_end = min(img_h, c_h + new_y)
        new_width = x_end - new_x
        new_height = y_end - new_y
        yolo_x = (new_x + new_width/2)/img_w
        yolo_y = (new_y + new_height/2)/img_h
        yolo_w = new_width/img_w
        yolo_h = new_height/img_h
        labels += [(component_label, yolo_x, yolo_y, yolo_w, yolo_h)]
        image[:,new_y:new_y+new_height,new_x:new_x+new_width] = np.add(image[:,new_y:new_y+new_height,new_x:new_x+new_width], component_image[:,:new_height,:new_width])
    
    def __getitem__(self, idx):
        image = torch.zeros(self.composite_scale)
        labels = []
        for component in self.get_components_to_add():
            self.add_component_to_image_and_labels(image, labels, component)
        #image = normalize_image(image, axis=-2)
        
        if self.transforms:
            if type(self.transforms) == list:
                for transform in self.transforms:
                    image = transform(image)
            else:
                image = self.transforms(image)
        #image = normalize_image(image, axis=-2)
        return image, labels