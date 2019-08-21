import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from sklearn.metrics import confusion_matrix
try:
    import matplotlib.pyplot as plt
except:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

class MyDataset(Dataset):
    def __init__(self, x, y, transform=None):
        self.features = x
        self.targets = y
        self.transforms = transform
        
    def __getitem__(self, index):
        x = self.features[index]
        y = self.targets[index]
        return (x, y)

    def __len__(self):
        return self.features.shape[0]

#32->64->64->128(dropout=.2)->20: 97.2%
#32->64->64->128(dropout=.5)->20: 87.7%
#32->64->64(dropout=.2)->20: 97.1%
class CNN_v1(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN_v1, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=4, stride=2, padding=1), #32x64x64
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(kernel_size=2, stride=2), #32x32x32
            )
        self.layer2 = nn.Sequential(
            nn.Conv2d(8, 4, kernel_size=3, stride=1, padding=1), #64x32x32
            nn.ReLU(),
            nn.BatchNorm2d(4),
            nn.MaxPool2d(kernel_size=2, stride=2) #32x16x16
            )
        self.layer3 = nn.Sequential(
            nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1), #64x32x32
            nn.ReLU(),
            nn.BatchNorm2d(4),
            #nn.MaxPool2d(kernel_size=2, stride=2) #32x16x16
            )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(in_features=4*8*8, out_features=48),
            nn.ReLU(),
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(in_features=48, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=num_classes)
        )
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        #out = self.layer3(out)
        #out = self.layer4(out)
        #out = self.layer5(out)
        out = out.reshape(out.size(0), -1)
        out = self.classifier(out)
        return out
    
    def converged(self, loss):
        if len(loss) < 3: return False
        else:
            return all(i < 0 for i in loss[-3:])


class DenseNet(nn.Module):
    def __init__(self, num_classes=10):
        super(DenseNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(in_features=64*64, out_features=512),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features=256, out_features=num_classes)
        )
    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        return out
    
    def converged(self, loss):
        if len(loss) < 3: return False
        else:
            return all(i < 0 for i in loss[-3:])


