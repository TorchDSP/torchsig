from torch import mean
from torch.nn import Module, LSTM

class DebugPrintLayer(Module):
    """
    A layer for debugging pytorch models; prints out the shape and data type of the input tensor at runtime
    returns he input tensor unchanged
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        print(x.shape, x.dtype)
        return x
    
class ScalingLayer(Module):
    """
    A layer that given input tensor x outputs scale_val * x
    used to linearly scale inputs by a fixed value
    """
    def __init__(self, scale_val):
        super().__init__()
        self.scale_val = scale_val

    def forward(self, x):
        return self.scale_val * x
    
class DropChannel(Module):
    """
    A layer that drops the last color channel of an image [must be in channel-first form]
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x[:,:-1,:,:]
    
class LSTMImageReader(Module):
    """
    TODO add some real documentation here
    """
    def __init__(self, input_width, lstm_width, img_shape, num_layers=2):
        super().__init__()
        self.img_shape = img_shape
        self.img_height = img_shape[0]
        self.img_width = img_shape[1]
        self.input_width = input_width
        self.lstm_width = lstm_width
        self.lstm_model = LSTM(self.input_width,self.lstm_width,num_layers,True,True,0,False,self.img_height)
        
    def forward(self, x):
        output, (h,c) = self.lstm_model(x.transpose(1,2))
        img_tensor = output.transpose(1,2)[:,:self.img_height,:self.img_width] #take only the last img_height entries in the outut sequence
        return img_tensor.reshape([x.size(0),1,self.img_height,self.img_width])

class Reshape(Module):
    """
    A layer that reshapes the input tensor to a tensor of the given shape
    if keep_batch_dim is True (defaults to True), the batch dimension is excluded from the reshape operation; otherwise it is included
    """
    def __init__(self, shape, keep_batch_dim=True):
        super(Reshape, self).__init__()
        self.shape = shape
        self.keep_batch_dim = keep_batch_dim

    def forward(self, x):
        if self.keep_batch_dim:
            batch_dim = x.size(0)
            shape = [batch_dim] + list(self.shape)
            return x.view(shape)
        return x.view(self.shape)
    
class Mean(Module):
    """
    A layer which returns the mean(s) along the dimension specified by dim of the input tensor
    """
    def __init__(self, dim):
        super(Mean, self).__init__()
        self.dim = dim

    def forward(self, x):
        return mean(x,self.dim)