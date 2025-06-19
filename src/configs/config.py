from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class GeneModelConfig:
    name: str = "default_config"
    k: int = 6 #K-Mers for the FCGR-Image
    pathogen: str = "Staphylococcus_aureus_cefoxitin"
    genes: List[str] = field(default_factory=lambda: [
        "dfrB", "fusA", "grlA", "grlB", "gyrA", "ileS", "pbp2", "pbp4-promoter", "pbp4", "rpoB"
    ])
    root_dir: str = "../data/ds1"
    batch_size: int = 16
    learning_rate: float = 0.0001
    epochs: int = 200
    rareclasssampling: bool = True #Makes sure that rare classes are primarily sampled so that they occur more often than their distribution allows it
    weight_decay: float = 0.01 
    trainvalsplit: float = 0.15 #Sets the percentage of training data that will be exclusively used for validation
    lossweighting: bool = True #Loss is weighted according to class distribution
    noise: Tuple[float, float] = (0.00, 0.03) #Noise minimum and maximum that is added to fcgr
    dropout: float = 0.5
    rareclasssamplerreplacement: bool = False #Rare classes can be drawn multiple times 
    returnlowestvalloss: bool = False #Returns the lowest validation loss model after training instead of the final
    onlyfcgr: bool = False #Uses only the FCGR-Image of the Gene-Sequence instead of FCGR-Image and the one-hot-encoded sequence
