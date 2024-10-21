import cv2
import numpy as np
import torch

def normalize_image(image, axis=None):
    """
    returns the infinity norm of an image
    Inputs:
        image: image to norm as a 2d ndarray
    Outputs:
        the normalized image
    """
    if type(image) != torch.Tensor:
            image = torch.Tensor(image)
    if axis == None:
        ans = image - image.min()
        return torch.clip(ans/max(ans.max(),0.0000001), 0, 1)
    else:
        ans = image - image.min(dim=axis, keepdim=True)[0]
        return torch.clip(ans/torch.clamp(ans.max(dim=axis, keepdim=True)[0],min=0.0000001), 0, 1)

def isolate_foreground_signal(image, filter_strength=0):
    '''
    filters image (a tensor of shape [1, width, height] in grayscale) to seperate foreground from background noise, and returns the filtered image tensor;
    an integer filter_strength can be passed in to tune the filtration effect
    '''
    test_hsv = cv2.cvtColor(cv2.cvtColor((image[0]*255).int().numpy().astype(np.uint8), cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2HSV)
    lower = np.array([0, 0, 0])
    upper = np.array([360, 255, int(255/2)]) # hand tuned, HARD CODED # TODO hard coded considered harmful :(
    upper = np.array([360, 255, int(255/2) - filter_strength]) # HARD CODED # TODO hard coded considered harmful :(

    mask = cv2.inRange(test_hsv, lower, upper)

    img_contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    blank = np.ones(image.shape[1:], np.uint8) * 255
    d = cv2.drawContours(blank, img_contours, -1, (0, 0, 0), -1)
    final_image = torch.Tensor(d[None,:,:]) * image

    return final_image