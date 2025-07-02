from dataclasses import dataclass, field
from typing import List
import numpy as np

@dataclass
class EpochMetrics:
    epoch: int = 0
    train_confusion: List[List[int]] = field(
        default_factory=lambda: [[0, 0], [0, 0]]
    )  # [[TN, FP], [FN, TP]]
    val_confusion: List[List[int]] = field(
        default_factory=lambda: [[0, 0], [0, 0]]
    )  # [[TN, FP], [FN, TP]]
    train_acc: float = 0.0
    train_loss: float = 0.0
    val_acc: float = 0.0
    val_loss: float = 0.0
    test_roc_fpr : np.ndarray = np.ndarray([])
    test_roc_tpr : np.ndarray = np.ndarray([])

    @property
    def train_tn(self) -> int:
        return self.train_confusion[0][0]

    @property
    def train_fp(self) -> int:
        return self.train_confusion[0][1]

    @property
    def train_fn(self) -> int:
        return self.train_confusion[1][0]

    @property
    def train_tp(self) -> int:
        return self.train_confusion[1][1]

    @property
    def val_tn(self) -> int:
        return self.val_confusion[0][0]

    @property
    def val_fp(self) -> int:
        return self.val_confusion[0][1]

    @property
    def val_fn(self) -> int:
        return self.val_confusion[1][0]

    @property
    def val_tp(self) -> int:
        return self.val_confusion[1][1]

    @property
    def train_tpr(self) -> float:
        denom = self.train_tp + self.train_fn
        return self.train_tp / denom if denom != 0 else 0.0

    @property
    def train_fpr(self) -> float:
        denom = self.train_fp + self.train_tn
        return self.train_fp / denom if denom != 0 else 0.0

    @property
    def val_tpr(self) -> float:
        denom = self.val_tp + self.val_fn
        return self.val_tp / denom if denom != 0 else 0.0

    @property
    def val_fpr(self) -> float:
        denom = self.val_fp + self.val_tn
        return self.val_fp / denom if denom != 0 else 0.0

    def print_train(self, num_epochs):
        print(self.train_confusion)
        print(self.val_confusion)
        print(
            f"Epoch {self.epoch+1}/{num_epochs} | "
            f"Train Loss: {self.train_loss:.4f} | "
            f"Train Acc: {self.train_acc:.4f} | "
            f"Val Loss: {self.val_loss:.4f} | "
            f"Val Acc: {self.val_acc:.4f} | "
        )

    def print_val(self):
        print(self.val_confusion)
        print(
            f"Test Results| "
            f"Test Loss: {self.val_loss:.4f} | "
            f"Test Acc: {self.val_acc:.4f}"
        )
