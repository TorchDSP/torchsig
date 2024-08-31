import os
import cv2 as cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from torchsig.image_datasets.transforms.impairments import normalize_image

def load_image_rgb(filepath):
    f = cv2.imread(filepath)
    img = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
    return img

def extract_bounding_boxes(filepath, filter_strength=None):
    return extract_bounding_boxes_from_image(load_image_rgb(filepath), filter_strength=filter_strength)

''' 
Get Coordinates, Height, Width of drawn bounding boxes
Inputs:
    img: image array in rbg format
Output:
    boxes: list of image arrays in rbg format, corresponding to the contents of each bounding box
'''
def extract_bounding_boxes_from_image(img, isolate=True, filter_strength=None):
    
    # color mask to get drawn bounding boxes
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    # H, S, V
    # masking values hand tuned to ignore grayscale pixels
    # (aka low saturation and value)
    lower = np.array([0, 70, 100]) # HARD CODED - TODO figure this out - no hard coding!
    upper = np.array([179, 255, 255]) # HARD CODED - TODO figure this out - no hard coding!

    mask = cv2.inRange(img_hsv, lower, upper)
    
    # find corners of boundiung boxes
    img_contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) # find image contours
    # get bounding rectangle
    numContours = len(img_contours)
    contours_poly = [None]*numContours
    boundRect = [None]*numContours
    for i,c in enumerate(img_contours):
        contours_poly[i] = cv2.approxPolyDP(c, 2, True)
        boundRect[i] = cv2.boundingRect(contours_poly[i])

    boxes = []
    upper_left_coords = []

    for i in range(numContours):
        x,y,w,h = boundRect[i]
        if (x,y) not in upper_left_coords: # remove duplicates
            pad = 1
            signal_image = img[y - pad:y + h + pad, x - pad:x + w + pad] 
            #return signal_image
            if isolate:
                if filter_strength == None:
                    filter_strength = 0
                signal_image = isolate_soi(signal_image, filter_strength)
            upper_left_coords.append((x,y))
            boxes.append(signal_image)
    
    return boxes

'''
Isolate SOI
soi_image, assumed BGR colorspace
'''
def isolate_soi(soi_image, filter_strength=0):
    test_hsv = cv2.cvtColor(soi_image, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 0, 0])
    upper = np.array([360, 255, int(255/2)]) # hand tuned, HARD CODED # TODO hard coded considered harmful :(
    upper = np.array([360, 255, int(255/2) - filter_strength]) # HARD CODED # TODO hard coded considered harmful :(

    mask = cv2.inRange(test_hsv, lower, upper)
    # plt.imshow(mask)

    img_contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    blank = np.ones(soi_image.shape, np.uint8) * 255
    d = cv2.drawContours(blank, img_contours, -1, (0, 0, 0), -1)
    
    final_image = np.bitwise_or(d, soi_image)

    return final_image

'''
Process SOI images
Inputs:
    filepaths: list of filepaths to SOI images
Outputs:
    boxes: list of extracted SOIs, list of SignalBoxImage
'''
def extract_sois(filepaths, filter_strength=None):
    # extract all SOI image patches
    boxes = []
    
    for filepath in filepaths:#tqdm(filepaths):
        f = cv2.imread(filepath)
        soi_boxes = extract_bounding_boxes(filepath, filter_strength=filter_strength)
        boxes += soi_boxes
        
    
    return boxes

"""
A Dataset class for loading marked signals of interest (SOIs) from images in a folder
Inputs:
    filepath: a string file path to a folder containing images in which all signals of interest have been marked wit ha colored bounding box
    transforms: either a single function or list of functions from images to images to be applied to each SOI; used for adding noise and impairments to data; defaults to None
    read_black_hot: whether or not to read loaded images as black-hot; this will invert the value of loaded SOIs
"""
class SOIExtractorDataset(Dataset):
    def __init__(self, filepath: str, transforms = None, read_black_hot = False, filter_strength = None):
        self.filepath = filepath
        self.transforms = transforms
        image_paths = []
        for f in os.listdir(filepath):
            if f.endswith(".png"):
                image_paths.append(os.path.join(filepath, f))
        if read_black_hot:
            self.sois = [normalize_image(-soid.mean(axis=-1)).unsqueeze(0) for soid in extract_sois(image_paths, filter_strength)]
        else:
            self.sois = [normalize_image(soid.mean(axis=-1)).unsqueeze(0) for soid in extract_sois(image_paths, filter_strength)]
        
    def __len__(self):
        return len(self.sois)
    def __getitem__(self, idx):
        soi = torch.Tensor(self.sois[idx])
        if self.transforms:
            if type(self.transforms) == list:
                for transform in self.transforms:
                    soi = transform(soi)
            else:
                soi = self.transforms(soi)
        return soi
    def next(self):
        return self[np.random.randint(len(self))]

"""
A Dataset class for loading image files from a directory
Inputs:
    filepath: a string file path to a folder containing .png images to load
    transforms: either a single function or list of functions from images to images to be applied to each loaded image; used for adding noise and impairments to data; defaults to None
    read_black_hot: whether or not to read loaded images as black-hot; this will invert the value of loaded SOIs
"""
class ImageDirectoryDataset(Dataset):
    def __init__(self, filepath: str, transforms = None, read_black_hot = False):
        self.filepath = filepath
        self.transforms = transforms
        image_paths = []
        for f in os.listdir(filepath):
            if f.endswith(".png"):
                image_paths.append(os.path.join(filepath, f))
        if read_black_hot:
            self.images = [normalize_image(-load_image_rgb(path).mean(axis=-1)).unsqueeze(0) for path in image_paths]
        else:
            self.images = [normalize_image(load_image_rgb(path).mean(axis=-1)).unsqueeze(0) for path in image_paths]
        
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        image = torch.Tensor(self.images[idx])
        if self.transforms:
            if type(self.transforms) == list:
                for transform in self.transforms:
                    image = transform(image)
            else:
                image = self.transforms(image)
        return image
    def next(self):
        return self[np.random.randint(len(self))]

"""
As ImageDirectoryDataset, but with lazy evaluation, so files are not loaded in advance
"""
class LazyImageDirectoryDataset(Dataset):
    def __init__(self, filepath: str, transforms = None, read_black_hot = False):
        self.filepath = filepath
        self.transforms = transforms
        self.image_paths = [os.path.join(filepath, f) for f in os.listdir(filepath) if f.endswith(".png")]
        self.read_black_hot = read_black_hot
        #if read_black_hot:
        #    self.images = [normalize_image(-load_image_rgb(path).mean(axis=-1)).unsqueeze(0) for path in image_paths]
        #else:
        #    self.images = [normalize_image(load_image_rgb(path).mean(axis=-1)).unsqueeze(0) for path in image_paths]
        
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, idx):
        image = []
        path = self.image_paths[idx]
        if self.read_black_hot:
            image = normalize_image(-load_image_rgb(path).mean(axis=-1)).unsqueeze(0)
        else:
            image = normalize_image(load_image_rgb(path).mean(axis=-1)).unsqueeze(0)
        image = torch.Tensor(image)
        if self.transforms:
            if type(self.transforms) == list:
                for transform in self.transforms:
                    image = transform(image)
            else:
                image = self.transforms(image)
        return image
    def next(self):
        return self[np.random.randint(len(self))]

