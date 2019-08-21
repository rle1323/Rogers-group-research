import os
import torch
import torchvision
from torch import nn
from torchvision import transforms

class ConvAE(nn.Module):
    def __init__(self):
        super(ConvAE, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=3, padding=1),  # b, 16, 22, 22
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 16, 11, 11
            nn.Conv2d(16, 8, kernel_size=3, stride=2, padding=1),  # b, 8, 6, 6
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2)  # b, 8, 3, 3
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, kernel_size=5, stride=3),  # b, 16, 11, 11
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 33, 33
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # b, 1, 64, 64
            nn.Tanh()
        )

        '''
        self.encoder = nn.Sequential(
            nn.Linear(64 * 64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(), 
            nn.Linear(64, 12), 
            nn.ReLU(), 
            nn.Linear(12, 3))
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(),
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(), 
            nn.Linear(128, 64 * 64), 
            nn.Sigmoid()
        )
        
    '''
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x