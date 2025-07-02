from torch import nn
import torch
import math

class OnlySequenceNet(nn.Module):
    def __init__(self,dropout=0.5,kernel_size = 3,k=7):
        super().__init__()
        
       
        # Raw sequence branch
        self.seq_cnn = nn.Sequential(
            nn.Conv1d(4, 16,7),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            nn.Conv1d(16, 32, 7),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.AdaptiveMaxPool1d(4),
            nn.Dropout(dropout),
            nn.Flatten()
        )
        
        # Combined classifier
        classifier_base_size = 4 * 32
        self.classifier = nn.Sequential(
            nn.Linear(classifier_base_size, 512),
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
        
        # Process sequence branch
        seq_features = self.seq_cnn(x_seq)
        #print(seq_features.shape)
        
        # Combine features
        return self.classifier(seq_features)