import torch.nn as nn
from torchvision import models

# import the model (inception v3 CNN)
Net = models.inception_v3(pretrained=True)

# Modifying the fully connected layers to fit dataset
Net.fc = nn.Sequential(
    nn.Linear(2048, 512),
    nn.ReLU(),
    nn.Dropout(p=0.25),
    nn.Linear(512, 64),
    nn.ReLU(),
    nn.Dropout(p=0.25),
    nn.Linear(64, 3)
)