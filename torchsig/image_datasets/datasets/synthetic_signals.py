import numpy as np
import torch
from torch.utils.data import Dataset

from torchsig.image_datasets.transforms.impairments import BlurTransform, normalize_image, pad_border

"""
A Dataset class for generating 2d greyscale image data
Inputs:
    generator_function: a function taking no arguments which returns a generated image
    class_id: the integer class id to associate with this image type; must be specified; generally should not be the same as other instances
    transforms: either a single function or list of functions from images to images to be applied to each generated image; used for adding noise and impairments to data; defaults to None
"""
class GeneratorFunctionDataset(Dataset):
    def __init__(self, generator_function, class_id: int, transforms = None):
        self.class_id = class_id
        self.generator_function = generator_function
        self.transforms = transforms
    def __len__(self):
        return 1 # this is somewhat arbitrary; it will generate as many instances as are asked for
    def __getitem__(self, idx):
        #image = normalize_image(self.generator_function())
        image = self.generator_function()
        label = self.class_id
        if self.transforms:
            if type(self.transforms) == list:
                for transform in self.transforms:
                    image = transform(image)
            else:
                image = self.transforms(image)
        return image, self.class_id
    def next(self):
        return self[0]


"""
Takes no arguments and returns a 2d ndarray representing the spectrogram of a randomly generated tone
Inputs:
    tone_width: the width of the tone to be generated; corresponds to the length in time of a simulated signal
Outputs:
    2d ndarray representing the spectrogram of a randomly generated tone
"""
def generate_tone(tone_width: int, max_height: int = 10, min_height: int = 3):
    height = np.random.randint(min_height, high=max_height+1)
    width = tone_width
    first_axis = torch.arange(height)*(3.141592653589792*2/height)
    second_axis = torch.ones(width)
    image = -torch.cos(torch.matmul(first_axis[:,None], second_axis[None,:])).unsqueeze(0)
    return image

"""
curried implementation of 'generate_tone'
"""
def tone_generator_function(tone_width: int, max_height: int = 10, min_height: int = 3):
    return lambda: generate_tone(tone_width, max_height=max_height, min_height=min_height)

"""
Takes no arguments and returns a 2d ndarray representing the spectrogram of a randomly generated 'signal' rectangle
Inputs:
    tone_width: the width of the tone to be generated; corresponds to the length in time of a simulated signal
Outputs:
    2d ndarray representing the spectrogram of a randomly generated signal
"""
def generate_rectangle_signal(min_width: int = 10, max_width: int = 100, max_height: int = 50, min_height: int = 5, use_blur=True):
    height = np.random.randint(min_height, high=max_height+1)
    width = np.random.randint(min_width, high=max_width+1)
    image = torch.ones([1,height,width])#.unsqueeze(0)
    #if use_blur:
    #    blur_transform = BlurTransform(blur_shape=max([x//4 for x in image.shape]), strength=1)
    #    image = blur_transform(image)
    return pad_border(image,1)

"""
curried implementation of 'generate_rectangle_signal'
"""
def rectangle_signal_generator_function(min_width: int = 10, max_width: int = 100, max_height: int = 50, min_height: int = 5, use_blur=True):
    return lambda: generate_rectangle_signal(min_width= min_width, max_width=max_width, max_height=max_height, min_height=min_height, use_blur=use_blur)

"""
Takes in a function which returns an image representing a signal, and returns that image repeated with a fixed offset
Inputs:
    generator_fn: the function called to produce the signal to repeat
    min_gap: the smallest allowable interval (in pixels) between signal repetitions
    max_gap: the largest allowable interval (in pixels) between signal repetitions
    repeat_axis: the axis over which the signal repeats
    min_repeats: the fewest repeats allowed
    max_repeats: the most repeasts allowed
Outputs:
    2d ndarray representing the spectrogram of a randomly generated signal
"""
def generate_repeated_signal(generator_fn, min_gap: int = 2, max_gap: int = 10, repeat_axis=-1, min_repeats=8, max_repeats=16):
    signal = generator_fn()
    gap = np.random.randint(min_gap, high=max_gap+1)
    n_repeats = np.random.randint(min_repeats, high=max_repeats+1)
    signal_length = list(signal.shape)[repeat_axis]
    total_length = (signal_length + gap)*n_repeats - gap
    pad_shape = [0,0]*len(signal.shape)
    pad_shape[repeat_axis*2 + 1] = gap
    padded_signal = torch.nn.functional.pad(signal,pad_shape[::-1])
    tile_shape = [1]*len(signal.shape)
    tile_shape[repeat_axis] = n_repeats - 1
    repeated_signal = torch.concat([signal, padded_signal.tile(tile_shape)], dim=repeat_axis)
    return torch.Tensor(repeated_signal)

"""
curried implementation of 'generate_repeated_signal'
"""
def repeated_signal_generator_function(generator_fn, min_gap: int = 2, max_gap: int = 10, repeat_axis=-1, min_repeats=30, max_repeats=50):
    return lambda: generate_repeated_signal(generator_fn, min_gap = min_gap, max_gap = max_gap, repeat_axis=repeat_axis, min_repeats=min_repeats, max_repeats=max_repeats)