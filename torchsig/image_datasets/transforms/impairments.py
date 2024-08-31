import torch
import numpy as np
import cv2


"""
returns the infinity norm of an image
Inputs:
    image: image to norm as a 2d ndarray
Outputs:
    the normalized image
"""

def normalize_image(image, axis=None):
    if type(image) != torch.Tensor:
            image = torch.Tensor(image)
    if axis == None:
        ans = image - image.min()
        return torch.clip(ans/max(ans.max(),0.0000001), 0, 1)
    else:
        ans = image - image.min(dim=axis, keepdim=True)[0]
        return torch.clip(ans/torch.clamp(ans.max(dim=axis, keepdim=True)[0],min=0.0000001), 0, 1)

def pad_border(image, to_pad):
    if type(image) != torch.Tensor:
        image = torch.Tensor(image)
    if type(to_pad) == int:
        to_pad = (to_pad,to_pad)
    pad_dims = []
    for dim in to_pad:
        pad_dims += [dim,dim]
    return torch.nn.functional.pad(image,pad_dims)

"""
returns a vertical gaussian norm of the image, so each vertical strip is normed based on it's standard deviaiton
Inputs:
    image: image to norm as a 2d ndarray
Outputs:
    the normalized image
"""
def scale_dynamic_range(image):
    stds = image.std(dim=-2, keepdim=True)
    stds_scaled = image/stds
    mean_stds = stds_scaled.mean(dim=-2, keepdim=True)
    mean_stds_scaled = image/mean_stds
    return mean_stds_scaled

"""
A class for adding gaussian noise to images; properties are set on init and used when the class is called on an image [e.g., my_instance(image) => transformed_image]
Inputs:
    mean: the mean of the noise distribution; defaults to 0
    std: the standard deviation of the noise distribution; defaults to 0.2
"""
class GaussianNoiseTransform():
    def __init__(self, mean: float=0, std: float=0.2):
        self.mean = mean
        self.std = std
    def __call__(self, image):
        noise = torch.normal(self.mean, self.std, image.shape)
        return normalize_image(image + noise)

"""
A class for scaling things
Inputs:
    scale: the scale factor to apply
"""
class ScaleTransform():
    def __init__(self, scale: float):
        self.scale = scale
    def __call__(self, image):
        return image * self.scale

"""
As GaussianNoiseTransform, but with random standard deviation selected uniformly from range
Inputs:
    mean: the mean of the noise distribution; defaults to 0
    range: the range of standard deviations to use for the noise distribution; defaults to (0.01, 0.5)
"""
class RandomGaussianNoiseTransform():
    def __init__(self, mean: float=0, range = (0.01, 0.5)):
        self.mean = mean
        self.range = range
    def __call__(self, image):
        std = np.random.uniform(self.range[0], self.range[1])
        noise = torch.normal(self.mean, std, image.shape)
        return normalize_image(image + noise)


"""
A class for adding ripple noise (simulating interference from multiple rf signals); properties are set on init and used when the class is called on an image [e.g., my_instance(image) => transformed_image]
Inputs:
    strength: the strength of the operation; new_image = (1 - strength)*old_image + (strength)*modified_image;
    num_emitors: the number of simulated emitors to use
"""
class RippleNoiseTransform():
    def __init__(self, strength, num_emitors = 30, image_shape = None, a = 1E-6, b = 1E-10, base_freq=100):
        self.num_emitors = num_emitors
        self.base_freq = base_freq
        self.a = torch.tensor([a])
        self.b = torch.tensor([b])
        self.strength = strength
        self.thetas = torch.linspace(0, 2*3.141592693589792, self.num_emitors)
        self.x_offsets = torch.cos(self.thetas) + .5
        self.y_offsets = torch.sin(self.thetas) + .5
        self.image_shape = image_shape
        self.r = None
        self.freq_exp = None
        if not image_shape == None:
            self.update_mesh_spacing(self.image_shape)

    "internal function used to pre-calculate mesh values to save time in __call__"
    def update_mesh_spacing(self, shape):
        x = torch.linspace(0, 1, shape[2])
        y = torch.linspace(0, 1, shape[1])
        
        xv, yv = torch.meshgrid(x, y, indexing='xy')
        
        xv_tile = torch.tile(xv.unsqueeze(2), (1, 1, len(self.thetas)))
        yv_tile = torch.tile(yv.unsqueeze(2), (1, 1, len(self.thetas)))
        self.r = torch.abs((xv_tile - self.x_offsets) + 1j*(yv_tile - self.y_offsets))
        self.freq_exp = torch.exp(3.14159j * self.r)
        
    def __call__(self, image): 
        if self.image_shape == None:
            self.update_mesh_spacing(image.shape)
        
        alphas = torch.exp(torch.log(self.a) * (torch.rand(len(self.thetas)) - torch.log(self.b)))
        freqs = self.base_freq * (torch.rand(len(self.thetas)) - 1.)
        freqs = self.freq_exp**freqs
        
        amps = torch.exp(-alphas * self.r)
        surf_amp = torch.abs(amps * freqs.real + 1)
        surf_amp /= torch.max(torch.abs(surf_amp))
        noise = torch.sum(surf_amp, -1).unsqueeze(0)
        
        return normalize_image(np.minimum((1 - self.strength)*image , self.strength*noise))

"""
A class for adding ripple noise as RippleNoiseGenerator, but with random strength
Inputs:
    range: the strength range of the operation in the form (min_strength, max_strength); strength will be selected uniformly on this range
    num_emitors: the number of simulated emitors to use
"""
class RandomRippleNoiseTransform():
    def __init__(self, range, num_emitors = 30, image_shape = None, a=1E-6, b=1E-10, base_freq = 100):
        self.num_emitors = num_emitors
        self.range = range
        self.ripple_transform = RippleNoiseTransform(0.1, num_emitors=self.num_emitors, image_shape=image_shape, a=a, b=b, base_freq=base_freq)
    def __call__(self, image):
        self.ripple_transform.strength = np.random.uniform(self.range[0],self.range[1])
        return self.ripple_transform(image)


"""
A class for adding gaussian blur to images; properties are set on init and used when the class is called on an image [e.g., my_instance(image) => transformed_image]
Inputs:
    strength: the strength of the operation; new_image = (1 - strength)*old_image + (strength)*modified_image; defaults to 1
    blur_shape: the size of the kernel over which to blur; either an tuple (x,y), or an int (in which case (x,x) is assumed); a larger kernel will blur more; defaults to 5
"""
class BlurTransform():
    def __init__(self, strength: float=1, blur_shape=5):
        self.strength = strength
        self.blur_shape = blur_shape
        if type(blur_shape) == int:
            self.blur_shape = (blur_shape, blur_shape)
        self.kernel = torch.ones([1,1] + list(self.blur_shape), dtype=torch.float32)
        self.kernel = self.kernel/self.kernel.sum()
    def __call__(self, image_raw):
        if type(image_raw) != torch.Tensor:
            image = torch.Tensor(image_raw)
        else:
            image = image_raw
        if len(image_raw.shape) < 3:
            image = image.unsqueeze(0)
        #image = 
        pad_size = np.max(self.blur_shape)
        kernel = self.kernel.repeat(image.shape[0],image.shape[0],1,1)
        blurred_image = torch.nn.functional.conv2d(pad_border(image,[pad_size,pad_size]), kernel, padding='same')[:,pad_size:-pad_size, pad_size:-pad_size]
        return (1 - self.strength)*image + self.strength*blurred_image


"""
A class for randomly resizing images; the minimum and maximum resize factors are specified on init, and applied when the class is called on an image [e.g., my_instance(image) => transformed_image]
Inputs:
    scale: a tuple of (minimum_x_scale_factor, maximum_x_scale_factor)
    y_scale: a tuple of (minimum_y_scale_factor, maximum_y_scale_factor); if None is provided, defaults to match scale
"""
class RandomImageResizeTransform():
    def __init__(self, scale, y_scale=None):
        self.min_x = float(scale[0])
        self.max_x = float(scale[1])
        if y_scale:
            self.min_y = float(y_scale[0])
            self.max_y = float(y_scale[1])
        else:
            self.min_y = self.min_x
            self.max_y = self.max_x
    def __call__(self, image):
        image = image.numpy().transpose(1,2,0)
        x_scale = np.random.uniform(self.min_x, self.max_x)
        y_scale = np.random.uniform(self.min_y, self.max_y)
        new_size = (int(image.shape[1]*x_scale), int(image.shape[0]*y_scale))
        img = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)
        if len(img.shape) < 3:
            img = img[:,:,None]
        return torch.Tensor(img.transpose(2,0,1))


