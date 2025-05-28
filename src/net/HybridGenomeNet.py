from torch import nn
import torch

class HybridGenomeNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        kernel_size = 3
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
        
        # Raw sequence branch
        self.seq_cnn = nn.Sequential(
            nn.Conv1d(4, 16, 5),
            nn.BatchNorm1d(16),
            nn.ReLU(),
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
        seq_features = self.seq_cnn(x_seq)
        #print(seq_features.shape)
        
        # Combine features
        combined = torch.cat([img_features, seq_features], dim=1)
        return self.classifier(combined)