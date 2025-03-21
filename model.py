import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from rich import print

class MultiModalNet(nn.Module):
    def __init__(self, num_metadata_features, num_classes=4):
        super(MultiModalNet, self).__init__()
        self.cnn = models.resnet18(pretrained=True)
        num_ftrs = self.cnn.fc.in_features
        self.cnn.fc = nn.Identity()
        
        self.meta_fc = nn.Sequential(
            nn.Linear(num_metadata_features, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(num_ftrs + 32, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x_img, x_meta):
        img_features = self.cnn(x_img)  
        print(img_features.shape)
        meta_features = self.meta_fc(x_meta)
        print(meta_features.shape)
        combined = torch.cat((img_features, meta_features), dim=1)
        out = self.classifier(combined)  # [batch_size, num_classes]
        
        return out
if __name__ == "__main__":
    model = MultiModalNet(num_metadata_features=10, num_classes=4)
    
    dummy_images = torch.randn(8, 3, 224, 224)  
    dummy_metadata = torch.randn(8, 10)
    
    output = model(dummy_images, dummy_metadata)
    print(output.shape)  # [8, 4]
