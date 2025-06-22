from typing import List, Literal, Tuple
from torch.utils.data import Dataset, DataLoader
from fastfcgr import FastFCGR
import numpy as np
from torchvision import transforms
from torchvision.transforms import functional as F
import torch

from configs.config import GeneModelConfig
from .amr_utility import load_gene_data

class AddUniformNoise:
    def __init__(self, low=-0.05, high=0.05):
        self.low = low
        self.high = high

    def __call__(self, tensor):
        noise = torch.empty_like(tensor).uniform_(self.low, self.high)
        return tensor + noise

class HybridGenomeDataset(Dataset):
    def __init__(
        self,
        config : GeneModelConfig,
        train_or_test : Literal["train", "test"]
    ):
        self.k = config.k
        self.sequences = []
        self.labels = []
        
        for gene in config.genes:
            pathogens = load_gene_data(config.root_dir, config.pathogen, gene)
            self.sequences.extend([x[1] for x in pathogens[train_or_test]])
            self.labels.extend([x[2] for x in pathogens[train_or_test]])
        
        self.max_seq_len = len(max(self.sequences, key=len))
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                AddUniformNoise(config.noise[0],config.noise[1]),
            ]
        )
        
    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx : int) -> Tuple[Tuple[torch.Tensor, np.ndarray], Literal[0, 1]]:
        seq : str = self.sequences[idx]

        # Generate FCGR image
        fcgr = FastFCGR()
        fcgr.initialize(k=self.k, isRNA=False)
        fcgr.set_sequence(seq)
        fcgr.calculate(scalingFactor=0.5)
        matrix: np.ndarray = fcgr.get_matrix

        img_data: np.ndarray = np.log2(matrix + 1)
        img_data = (img_data / img_data.max() * 255).astype(np.float32)
        fcgr_image: torch.Tensor = self.transform(img_data)

        # Generate one-hot encoded sequence
        seq_encoded: np.ndarray = self.one_hot_encode(seq)

        return (fcgr_image, seq_encoded), self.labels[idx]

    def one_hot_encode(self, seq: str) -> torch.FloatTensor:
        mapping: dict[str, List[int]] = {
            "A": [1, 0, 0, 0],
            "T": [0, 1, 0, 0],
            "C": [0, 0, 1, 0],
            "G": [0, 0, 0, 1],
        }

        encoded: List[List[int]] = []
        for base in seq[: self.max_seq_len]:
            encoded.append(mapping.get(base.upper(), [0, 0, 0, 0]))

        if len(encoded) < self.max_seq_len:
            encoded += [[0, 0, 0, 0]] * (self.max_seq_len - len(encoded))
        else:
            encoded = encoded[: self.max_seq_len]

        return torch.FloatTensor(encoded).permute(1, 0)  # shape: (4, seq_len)