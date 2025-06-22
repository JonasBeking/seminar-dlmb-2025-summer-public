import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split,WeightedRandomSampler
from collections import Counter
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np

from amr.dataset import HybridGenomeDataset
from configs.config import GeneModelConfig

def hybrid_collate(batch):
    images = []
    sequences = []
    labels = []
    
    for (img, seq), label in batch:
        images.append(img)
        sequences.append(seq)
        labels.append(label)
    
    return (torch.stack(images), pad_sequence(sequences, batch_first=True)), F.one_hot(torch.tensor(labels), num_classes=2)


def get_dataloader(dataset,batch_size,sampler=None):
    return DataLoader(dataset,batch_size=batch_size,collate_fn=hybrid_collate,sampler=sampler)

def get_random_sampler(subset : Subset,replacement : bool):
    labels = [subset.dataset[i][1] for i in subset.indices]  # get labels of the subset

    # 2. Count occurrences of each class
    label_counts = Counter(labels)
    total_count = sum(label_counts.values())

    # 3. Compute weight for each class: inverse frequency
    class_weights = {label: total_count / count for label, count in label_counts.items()}

    # 4. Assign weight to each sample in the subset
    sample_weights = [class_weights[label] for label in labels]

    # 5. Create WeightedRandomSampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(subset),
        replacement=replacement
    )
    
    return sampler,class_weights

def get_train_val_dataloaders(config : GeneModelConfig):
    dataset = HybridGenomeDataset(
        config=config,
        train_or_test="train"
    )
    batch_size = config.batch_size
    val_split = config.trainvalsplit
    rareclasssampling = config.rareclasssampling
    replacement = config.rareclasssamplerreplacement
    
    labels = dataset.labels  # For ImageFolder datasets   
    # For custom datasets: labels = [label for _, label in dataset]

    # Split indices stratify=True ensures preserved class ratios
    train_indices, val_indices = train_test_split(
        range(len(dataset)),  # List of all indices
        test_size=val_split,        # 1:4 ratio (20% validation)
        stratify=labels,      # Use the actual labels from your dataset
        random_state=42
    )

    # Create subsets
    train_split_dataset = Subset(dataset, train_indices)
    val_split_dataset = Subset(dataset, val_indices)
    
    train_sampler,class_weights_train = get_random_sampler(train_split_dataset,replacement=replacement)
    val_sampler,class_weights_val = get_random_sampler(val_split_dataset,replacement=replacement)

    train_loader = get_dataloader(train_split_dataset,batch_size,sampler=train_sampler if rareclasssampling else None)
    val_loader = get_dataloader(val_split_dataset,batch_size,sampler=val_sampler if rareclasssampling else None)
    return train_loader,val_loader,class_weights_train

def get_test_dataloader(config : GeneModelConfig):
    test_dataset = HybridGenomeDataset(
        config=config,
        train_or_test="test"
    )
    test_loader = get_dataloader(test_dataset, config.batch_size)
    return test_loader