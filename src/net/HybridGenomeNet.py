from torch import nn
import torch

class HybridGenomeNet(nn.Module):
    def __init__(self, num_classes=2, seq_length=5000):
        super().__init__()
        
        # FCGR image branch
        self.image_cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )
        
        # Raw sequence branch
        self.seq_cnn = nn.Sequential(
            nn.Conv1d(4, 16, 5),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 5),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        
        # Combined classifier
        self.classifier = nn.Sequential(
            nn.Linear(64*(64//4)**2 + 32 + 1, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(512,32),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        x_img, x_seq,x_gene = x
        
        # Process image branch
        img_features = self.image_cnn(x_img)
        
        # Process sequence branch
        seq_features = self.seq_cnn(x_seq)
        
        # Combine features
        combined = torch.cat([img_features, seq_features,x_gene], dim=1)
        return self.classifier(combined)