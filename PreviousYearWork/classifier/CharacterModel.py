import torch.nn as nn
import torchvision


class CharacterModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.resnet50(weights='IMAGENET1K_V1')
        self.num_filters = self.model.fc.in_features  # 2048 for resnet18
        self.fc = nn.Sequential(
            nn.Linear(self.num_filters, 512),        
            nn.ReLU(),
            nn.BatchNorm1d(512),             
            nn.Dropout(0.5),                 
            nn.Linear(512, 35)      
        )
        self.model.fc = self.fc
    
    def forward(self, x):
        return self.model(x)