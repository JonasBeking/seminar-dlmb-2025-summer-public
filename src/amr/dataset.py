from torch.utils.data import Dataset, DataLoader
from fastfcgr import FastFCGR
import numpy as np
from torchvision import transforms
from torchvision.transforms import functional as F
import torch
from .amr_utility import load_gene_data
import random


class HybridGenomeDataset(Dataset):
    def __init__(
        self,
        root_dir,
        train_or_test="train",
        pathogen="Staphylococcus_aureus_cefoxitin",
        genes=["pbp4"],
        k=6,
        img_size=64,
        max_seq_len=5000,
        
    ):
        self.root_dir = root_dir
        self.k = k
        self.img_size = img_size
        self.max_seq_len = max_seq_len
        
        self.sequences = []
        self.labels = []
        self.genes = []
        
        
        
        for idx,gene in enumerate(genes):
            pathogens = load_gene_data(self.root_dir, pathogen, gene)
            self.sequences.extend([x[1] for x in pathogens[train_or_test]])
            self.labels.extend([x[2] for x in pathogens[train_or_test]])
            self.genes.extend([[idx + 1] for x in pathogens[train_or_test]])
        

        
        self.max_seq_len = len(max(self.sequences, key=len))
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
                transforms.Resize((img_size, img_size), antialias=True),
            ]
        )

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        gene = self.genes[idx]

        # Generate FCGR image
        fcgr = FastFCGR()
        fcgr.initialize(k=self.k, isRNA=False)
        fcgr.set_sequence(seq)
        fcgr.calculate(scalingFactor=0.5)
        matrix = fcgr.get_matrix
        img_data = np.log2(matrix + 1)
        img_data = (img_data / img_data.max() * 255).astype(np.uint8)
        fcgr_image = self.transform(img_data)

        # Generate one-hot encoded sequence
        seq_encoded = self.one_hot_encode(seq)

        return (fcgr_image, seq_encoded,gene), self.labels[idx]

    def one_hot_encode(self, seq):
        mapping = {
            "A": [1, 0, 0, 0],
            "T": [0, 1, 0, 0],
            "C": [0, 0, 1, 0],
            "G": [0, 0, 0, 1],
        }
        encoded = []
        for base in seq[: self.max_seq_len]:
            encoded.append(mapping.get(base.upper(), [0, 0, 0, 0]))
        # Pad/truncate to fixed length
        if len(encoded) < self.max_seq_len:
            encoded += [[0, 0, 0, 0]] * (self.max_seq_len - len(encoded))
        else:
            encoded = encoded[: self.max_seq_len]
        return torch.FloatTensor(encoded).permute(1, 0)  # (4, seq_len)
