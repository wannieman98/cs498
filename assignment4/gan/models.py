import torch
import torch.nn as nn
from .spectral_normalization import SpectralNorm

class Discriminator(torch.nn.Module):
    def __init__(self, input_channels=3):
        super(Discriminator, self).__init__()
        
        #Hint: Hint: Apply spectral normalization to convolutional layers. Input to SpectralNorm should be your conv nn module
        ####################################
        #          YOUR CODE HERE          #
        ####################################
        self.discriminator = nn.Sequential(
            SpectralNorm(nn.Conv2d(input_channels, 128, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            SpectralNorm(nn.Conv2d(128, 256, 4, 2, 1)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            SpectralNorm(nn.Conv2d(256, 512, 4, 2, 1)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            SpectralNorm(nn.Conv2d(512, 1024, 4, 2, 1)),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            SpectralNorm(nn.Conv2d(1024, 1, 4, 1, 1)),
            nn.LeakyReLU(0.2),
        )
        ##########       END      ##########
    
    def forward(self, x):
        
        return self.discriminator(x)


class Generator(torch.nn.Module):
    def __init__(self, noise_dim, output_channels=3):
        super(Generator, self).__init__()
        
        ####################################
        #          YOUR CODE HERE          #
        ####################################
        self.noise_dim = noise_dim
        self.generator = nn.Sequential(
            nn.ConvTranspose2d(noise_dim, 1024, 4, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, output_channels, 4, 2, 1),
            nn.ReLU(),
            nn.Tanh()
        )

        ##########       END      ##########
    
    def forward(self, x):
        
        x = x.view(-1, self.noise_dim, 1, 1)
        return self.generator(x)
    

