import torch
import torch.nn as nn
from torch.nn.modules.activation import Tanh
from .spectral_normalization import SpectralNorm

class Discriminator(torch.nn.Module):
    def __init__(self, input_channels=3):
        super(Discriminator, self).__init__()
        
        #Hint: Hint: Apply spectral normalization to convolutional layers. Input to SpectralNorm should be your conv nn module
        ####################################
        #          YOUR CODE HERE          #
        ####################################
        self.layer_1 = nn.Sequential(
            SpectralNorm(nn.Conv2d(3, 128, 4, 2, 1)),
            nn.LeakyReLU(0.2)
        )
        self.layer_2 = nn.Sequential(
            SpectralNorm(nn.Conv2d(128, 256, 4, 2, 1)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)
        )
        self.layer_3 = nn.Sequential(
            SpectralNorm(nn.Conv2d(256, 512, 4, 2, 1)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2)
        )
        self.layer_4 = nn.Sequential(
            SpectralNorm(nn.Conv2d(512, 1024, 4, 2, 1)),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2)
        )
        self.layer_5 = nn.Sequential(
            SpectralNorm(nn.Conv2d(1024, 1, 4, 1, 1)),
            nn.LeakyReLU(0.2)
        )
        ##########       END      ##########
    
    def forward(self, x):
        
        ####################################
        #          YOUR CODE HERE          #
        ####################################
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)
        ##########       END      ##########
        
        return x


class Generator(torch.nn.Module):
    def __init__(self, noise_dim, output_channels=3):
        super(Generator, self).__init__()    
        
        ####################################
        #          YOUR CODE HERE          #
        ####################################
        self.layer_1 = nn.Sequential(
            nn.ConvTranspose2d(noise_dim, 1024, 4, 1, 0),
            nn.BatchNorm2d(1024),
            nn.ReLU()
        )
        self.layer_2 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.layer_3 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.layer_4 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.layer_5 = nn.Sequential(
            nn.ConvTranspose2d(128, 3, 4, 2, 1),
            nn.ReLU(),
            Tanh()
        )
        ##########       END      ##########
    
    def forward(self, x):
        
        ####################################
        #          YOUR CODE HERE          #
        ####################################
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)
        ##########       END      ##########
        
        return x
    

