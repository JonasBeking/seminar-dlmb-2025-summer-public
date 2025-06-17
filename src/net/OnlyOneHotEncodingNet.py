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

class OnlyOneHotEncodingNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        kernel_size_staphy = 3
        kernel_size_kleb = 3
        kernel_size = kernel_size_staphy
        padding = int((kernel_size - 1) / 2)
        dropout = 0.1
        
        # Raw sequence branch
        self.seq_cnn = nn.Sequential(
            Unsqueeze(1),
            nn.Conv2d(1, 16, 4),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            Squeeze(-2),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            nn.Conv1d(16, 32, 5),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
            nn.Dropout(dropout),
            nn.Flatten()
        )
        
        # Combined classifier
        self.classifier = nn.Sequential(
            nn.Linear(32, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x_img, x_seq = x
    
        # Process sequence branch
        seq_features = self.seq_cnn(x_seq)
        return self.classifier(seq_features)