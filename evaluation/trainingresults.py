from dataclasses import dataclass, field
from typing import List, Any
import numpy as np

from configs.config import GeneModelConfig  # if your confusion matrix is a NumPy array

@dataclass
class TrainingResults:
    gene: str
    accs: List[float] = field(default_factory=list)
    losses: List[float] = field(default_factory=list)
    tpr: List[float] = field(default_factory=list)
    fpr: List[float] = field(default_factory=list)
    val_accs: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    val_tpr: List[float] = field(default_factory=list)
    val_fpr: List[float] = field(default_factory=list)
    test_loss: float = 0.0
    test_acc: float = 0.0
    confusion_matrix: Any = None  # e.g., a 2D list or np.ndarray
    config: GeneModelConfig = None  # GeneModelConfig or a dict, depending on use