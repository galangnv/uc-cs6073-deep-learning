import torch.nn as nn

class Reshape(nn.Module):

    def __init__(self, *args):
        super().__init__()
        self.shape = args
    
    def forward(self, x):
        return x.view(self.shape)

class Trim(nn.Module):
    
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size
    
    def forward(self, x):
        return x[:, :, :self.output_size, :self.output_size]