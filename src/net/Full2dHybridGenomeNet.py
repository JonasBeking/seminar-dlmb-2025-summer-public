from torch import nn
import torch

class Squeeze(nn.Module):
    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.squeeze(dim=self.dim) if self.dim is not None else x.squeeze()
    
class Unsqueeze(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.unsqueeze(dim=self.dim)

class Full2dHybridGenomeNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        kernel_size_staphy = 3
        kernel_size_kleb = 3
        kernel_size = kernel_size_staphy
        padding = int((kernel_size - 1) / 2)
        dropout = 0.5
        
        
        
        # FCGR image branch
        self.image_cnn = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size, padding=padding),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(dropout),
            nn.Conv2d(128, 64, kernel_size, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(dropout),
            nn.Flatten()
        )
        
        self.first_seq_cnn = nn.Sequential(
            Unsqueeze(1),
            nn.Conv2d(1, 16, (4,16)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            Squeeze(-2),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            nn.Conv1d(16, 32, 5),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.AdaptiveAvgPool1d(1),
            nn.Dropout(dropout),
            nn.Flatten()
        )
        
        # Combined classifier
        self.classifier = nn.Sequential(
            nn.Linear(16416, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(32, 2)
        )

    def forward(self, x):
        x_img, x_seq = x
        
        # Process image branch
        img_features = self.image_cnn(x_img)
        #print(img_features.shape)
        
        # Process sequence branch
        #x_seq = x_seq.unsqueeze(1)

        seq_features = self.first_seq_cnn(x_seq)
        
        # Combine features
        combined = torch.cat([img_features, seq_features], dim=1)
        return self.classifier(combined)