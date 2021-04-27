import torch 
import torch.nn as nn
from torch.nn.modules.activation import Tanh
from .spectral_normalization import SpectralNorm

class Generator(torch.nn.Module):
    def __init__(self, input_channels=3):
        super(Generator, self).__init__()
