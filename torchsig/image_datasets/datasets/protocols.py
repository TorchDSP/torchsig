from torch import Tensor
from torch.utils.data import Dataset
import numpy as np
from torchsig.image_datasets.datasets.yolo_datasets import YOLODatum

"""
A Dataset class for combining signals at different frequencies (y-axis in spectrogram image tensor) to simulate frequency hopping; 
the grammar will be empty at initialization, and production rules must be added using the 'add_rule' function
Inputs:
    signal_fn: the the function to call to generate signals to use; will be called each time a signal is needed; if a dataset is passed in, a random element of the dataset will be used instead
    channel_height: the height of the channel y axis in pixels; corresponds to channel bandwidth; total height will be channel_height * num_channels; can be a single number or random range (low,high)
    num_channels: the number of channels used in transmission; can be a single number or random range (low,high)
    signal_length: the length of each signal time slot (length on x amis of spectrogram image); signals beyond this length will be truncated; can be a single number or random range (low,high)
    num_signals: the number of signals to generate; either an int or a 2 element tuple of the form (lower, upper), in which case a random number will be selected between lower and upper (inclusive) at runtime
    hopping_function: the function used for determining the new channel based on the previous channel and total number of channels; if none, the next channel down will be selected each time (wrapping at the bottom)
    transforms: either a single function or list of functions from images to images to be applied to each generated image; used for adding noise and impairments to data; defaults to None
"""
class FrequencyHoppingDataset(Dataset):
    def __init__(self, signal_fn, channel_height: int, num_channels: int, signal_length: int, num_signals, hopping_function = None, transforms = None):
        self.signal_fn = signal_fn
        if isinstance(signal_fn, Dataset):
            self.signal_fn = lambda: signal_fn[np.random.randint(len(signal_fn))]
        self.channel_height = channel_height
        self.num_channels = num_channels
        self.signal_length = signal_length
        self.num_signals = num_signals
        if type(self.num_signals) == int:
            self.num_signals = (num_signals, num_signals)
        if type(self.signal_length) == int:
            self.signal_length = (signal_length, signal_length)
        if type(self.num_channels) == int:
            self.num_channels = (num_channels, num_channels)
        if type(self.channel_height) == int:
            self.channel_height = (channel_height, channel_height)
        self.hopping_function = hopping_function
        self.transforms = transforms
    def __len__(self):
        return 1 # this is somewhat arbitrary; it will generate as many instances as are asked for
    def __getitem__(self, idx):
        image = self.generate_hopping_signal()
        if self.transforms:
            if type(self.transforms) == list:
                for transform in self.transforms:
                    image = transform(image)
            else:
                image = self.transforms(image)
        return image

    """
    A method that generates a new signal by applying channel hopping
    Outputs:
        the generated spectrogram image tensor
    """
    def generate_hopping_signal(self):
        num_signals = np.random.randint(self.num_signals[0], self.num_signals[1]+1)
        signal_length = np.random.randint(self.signal_length[0], self.signal_length[1]+1)
        num_channels = np.random.randint(self.num_channels[0], self.num_channels[1]+1)
        channel_height = np.random.randint(self.channel_height[0], self.channel_height[1]+1)
        subsignals = [self.signal_fn() for i in range(num_signals)]
        img = self.format_blank_image(np.zeros([1,channel_height*num_channels, signal_length*len(subsignals)]))
        channel_order = []
        if self.hopping_function == None:
            init_channel = np.random.randint(0, num_channels+1)
            channel_order = (list(range(num_channels))*((len(subsignals)//num_channels)+2))[init_channel:len(subsignals)+init_channel] #cycle through the channels in order from random start channel
            if np.random.randint(2) == 1:
                channel_order = channel_order[::-1]
        else:
            for i in range(len(subsignals)):
                channel_order += [self.hopping_function(num_channels, channel_order)]
        for i in range(len(subsignals)):
            subsignal = subsignals[i]
            channel = channel_order[i]
            start_x = signal_length*i
            start_y = channel_height*channel
            width = min(signal_length, subsignal.size(2))
            height = min(channel_height, subsignal.size(1))
            #subsignal = subsignal[:, :height, :width] # cap the height and width
            self.compose_data(img, subsignal, (start_x, start_y))
        return img

    """
    A method for turning a blank image of the correct shape to a valid datum to be returned by the dataset; trivial in this class;
    Can be overriden to maintain more complex data formats in subclasses
    """
    def format_blank_image(self, img):
        return Tensor(img)

    """
    A method for combining data from parts of the final image into the whole
    Can be overridden in subclasses to change behavior
    """
    def compose_data(self, img, subsignal, top_left):
        _, height, width = subsignal.shape
        start_x, start_y = top_left
        img[:, start_y:start_y+height, start_x:start_x+width] = subsignal[:,:height,:width]
    
    def next(self):
        return self[0]

"""
As FrequencyHoppingDataset, but for handling YOLO data. Will combine signals in such a way as to maintain YOLO formatted bounding box labels around each signal
"""
class YOLOFrequencyHoppingDataset(FrequencyHoppingDataset):
    def __init__(self, signal_fn, channel_height: int, num_channels: int, signal_length: int, num_signals, hopping_function = None, transforms = None):
        FrequencyHoppingDataset.__init__(self, signal_fn, channel_height, num_channels, signal_length, num_signals, hopping_function, transforms)
    """
    A turn blank image into blank YOLODatum
    """
    def format_blank_image(self, img):
        return YOLODatum(Tensor(img), [])
    """
    compose YOLO data
    """
    def compose_data(self, img, subsignal, top_left):
        if not isinstance(subsignal, YOLODatum):
            subsignal = YOLODatum(subsignal, [])
        img.compose_yolo_data(subsignal, top_left)


"""
A Dataset class for generating signals based on user defined signal protocols written as context free grammar (CFG); 
the grammar will be empty at initialization, and production rules must be added using the 'add_rule' function
Inputs:
    transforms: either a single function or list of functions from images to images to be applied to each generated image; used for adding noise and impairments to data; defaults to None
"""
class CFGSignalProtocolDataset(Dataset):
    def __init__(self, initial_token:str = None, transforms = None):
        self.rules = {}
        self.initial_token = initial_token
        self.transforms = transforms
    def __len__(self):
        return 1 # this is somewhat arbitrary; it will generate as many instances as are asked for
    def __getitem__(self, idx):
        image = self.get_random_product()
        if self.transforms:
            if type(self.transforms) == list:
                for transform in self.transforms:
                    image = transform(image)
            else:
                image = self.transforms(image)
        return image

    """
    A method that adds a new production rule to the dataset's CFG. The rules added will be used to generate new random CFG products when the dataset is accessed
    Inputs:
        token: the string token on the left hand side of hte production rule; products containing this token will be evaluated using the new rule
        product: a list containing some number of strings and/or functions; the token will be evaluated to product; 
            each string in product will be further evaluated as a token in the CFG; each function in product will be called without argument, and should return an image tensor
        priority: the likelihood of using this rule as opposed to other rules for the same token during evaluation;
            expressed as a float, such that a rule with twice as high a priority will be used twice as often
    """
    def add_rule(self, token: str, product:list, priority:float=1):
        if not type(product) is list:
            product = [product]
        for i in range(len(product)):
            if isinstance(product[i], Dataset):
                temp_ds = product[i]
                product[i] = lambda: temp_ds[np.random.randint(len(temp_ds))]
        for e in product:
            if not type(e) in [str, type(lambda x: 4), type(None)]: 
                #check is everything is a string or a function; if not throw an error
                raise(Exception("Invalid production rule given; all products of a production rule must be None, of type string, or of type function, but was given:" + str(type(e))))
        if not token in self.rules.keys():
            self.rules[token] = []
        self.rules[token] += [(priority, product)]

    """
    A method that sets the initial token of the CFG to a given string
    Inputs:
        token: the string token to be set as the initial token
    """
    def set_initial_token(self, token: str):
       self.initial_token = token

    """
    A method that used the production rules of the CFG to generate a new composite image
    Outputs:
        the image generated
    """
    def get_random_product(self):
        return self.combine_products(self.get_subproduct_list())
    def combine_products(self, list_of_subproducts):
        heights = []
        width = 0
        for subproduct in list_of_subproducts:
            heights += [subproduct.size(1)]
            width += subproduct.size(2)
        height = np.max(heights)
        new_img = self.format_blank_image(np.zeros([1,height,width]))

        prev_x = 0
        mid_y = height//2
        for subproduct in list_of_subproducts:
            low_y = mid_y - subproduct.size(1)//2
            new_x = prev_x + subproduct.size(2)
            self.compose_data(new_img, subproduct, (prev_x, low_y))
            prev_x = new_x
        return new_img
    """
    A method for turning a blank image of the correct shape to a valid datum to be returned by the dataset; trivial in this class;
    Can be overriden to maintain more complex data formats in subclasses
    """
    def format_blank_image(self, img):
        return Tensor(img)
    """
    A method for combining data from parts of the final image into the whole
    Can be overridden in subclasses to change behavior
    """
    def compose_data(self, img, subsignal, top_left):
        _, height, width = subsignal.shape
        start_x, start_y = top_left
        img[:, start_y:start_y+height, start_x:start_x+width] = subsignal[:,:height,:width]
    
    def get_subproduct_list(self):
        if self.initial_token == None:
            raise(Exception("No initial token was given to the CFG"))
        return [f() for f in self.get_token_product(self.initial_token) if not f == None]
    def get_token_product(self, token):
        rules = self.rules.copy()[token]
        if len(rules) < 1:
            raise(Exception("No production rules exist to resolve token: '"+token+"'"))
        probabilities = []
        product_lists = []
        for rule in rules:
            if rule[0] < 0:
                raise(Exception("Rule priorities cannot be negative"))
            probabilities += [rule[0]]
            product_lists += [rule[1]]
        probabilities = np.array(probabilities)/np.sum(probabilities)
        sub_product = product_lists[np.random.choice(len(product_lists), p=probabilities)].copy()
        for i in range(len(sub_product)):
            if type(sub_product[i]) is str:
                sub_product[i] = self.get_token_product(sub_product[i])
        token_product = []
        for e in sub_product:
            if type(e) is list:
                token_product += e
            else:
                token_product += [e]
        return token_product.copy()
            
    
    
    def next(self):
        return self[0]


"""
As CFGSignalProtocolDataset, but signals are combined from top to bottom rather than from left to right; allows for different shaped of combined signal to be composed
"""
class VerticalCFGSignalProtocolDataset(CFGSignalProtocolDataset):
    def __init__(self, initial_token:str = None, transforms = None):
        CFGSignalProtocolDataset.__init__(self, initial_token, transforms)
        
    def combine_products(self, list_of_subproducts):
        widths = []
        height = 0
        for subproduct in list_of_subproducts:
            widths += [subproduct.size(2)]
            height += subproduct.size(1)
        width = np.max(widths)
        new_img = self.format_blank_image(np.zeros([1,height,width]))

        prev_y = 0
        mid_x = width//2
        for subproduct in list_of_subproducts:
            low_x = mid_x - subproduct.size(2)//2
            new_y = prev_y + subproduct.size(1)
            self.compose_data(new_img, subproduct, (low_x, prev_y))
            prev_y = new_y
        return new_img

"""
As VerticalCFGSignalProtocolDataset, but for handling YOLO data. Will combine signals in such a way as to maintain YOLO formatted bounding box labels around each signal
"""
class YOLOCFGSignalProtocolDataset(CFGSignalProtocolDataset):
    def __init__(self, initial_token:str = None, transforms = None):
        CFGSignalProtocolDataset.__init__(self, initial_token, transforms)
    """
    A turn blank image into blank YOLODatum
    """
    def format_blank_image(self, img):
        return YOLODatum(Tensor(img), [])
    """
    compose YOLO data
    """
    def compose_data(self, img, subsignal, top_left):
        if not isinstance(subsignal, YOLODatum):
            subsignal = YOLODatum(subsignal, [])
        img.compose_yolo_data(subsignal, top_left)

"""
As CFGSignalProtocolDataset, but for handling YOLO data. Will combine signals in such a way as to maintain YOLO formatted bounding box labels around each signal
"""
class YOLOVerticalCFGSignalProtocolDataset(VerticalCFGSignalProtocolDataset):
    def __init__(self, initial_token:str = None, transforms = None):
        VerticalCFGSignalProtocolDataset.__init__(self, initial_token, transforms)
    """
    A turn blank image into blank YOLODatum
    """
    def format_blank_image(self, img):
        return YOLODatum(Tensor(img), [])
    """
    compose YOLO data
    """
    def compose_data(self, img, subsignal, top_left):
        if not isinstance(subsignal, YOLODatum):
            subsignal = YOLODatum(subsignal, [])
        img.compose_yolo_data(subsignal, top_left)


def random_hopping(n_channels, channel_order):
    new_channel = np.random.randint(n_channels)
    if len(channel_order) == 0:
        return new_channel
    while channel_order[-1] == new_channel:
        new_channel = np.random.randint(n_channels)
    return new_channel

