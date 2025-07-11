from .config import GeneModelConfig

dfrB_config = GeneModelConfig(
    name="dfrB",
    genes=["dfrB"],
    dropout=0.15,
    noise=(0,0.3),
    batch_size=32,
    learning_rate=0.0001
)
fusA_config = GeneModelConfig(
    name="fusA",
    genes=["fusA"],
    dropout=0.05,
)
grlA_config = GeneModelConfig(
    name="grlA",
    genes=["grlA"],
    dropout=0.15,
    noise=(0.0, 0.1),
    batch_size=32,
    learning_rate=0.001
)
grlB_config = GeneModelConfig(
    name="grlB",
    genes=["grlB"],
    learning_rate=0.0001,
    noise=(0, 0.1),
    dropout=0.2,
    trainvalsplit=0.15,
    batch_size=8
)
gyrA_config = GeneModelConfig(
    name="gyrA",
    genes=["gyrA"],
    dropout=0.1,
    noise=(0.0, 0.01),
    weight_decay=0.2,
    trainvalsplit=0.2
)
ileS_config = GeneModelConfig(
    k=6,
    name="ileS",
    genes=["ileS"],
    batch_size=64,
    dropout=0.3,
    trainvalsplit=0.2,
    learning_rate=0.001,
    noise=(0.00,0.01)
)
pbp2_config = GeneModelConfig(
    name="pbp2",
    genes=["pbp2"],
    dropout=0.4,
    trainvalsplit=0.15,
    learning_rate=0.001,
    noise=(0.00,0.0),
    weight_decay=0.05,
    batch_size=32
)
pbp4_promoter_config = GeneModelConfig(
    name="pbp4_promoter",
    genes=["pbp4-promoter"],
    trainvalsplit=0.15,
    dropout=0.1,
    learning_rate=0.0001,
    weight_decay=0.9,
    noise=(0.00,3.0),
    batch_size=16
)
pbp4_config = GeneModelConfig(
    name="pbp4",
    genes=["pbp4"],
    batch_size=114,
    rareclasssampling=False,
    learning_rate=0.0001,
    dropout=0.1,
    noise=(0, 0.15),
    trainvalsplit=0.15,
)
rpoB_config = GeneModelConfig(
    name="rpoB",
    genes=["rpoB"],
    batch_size=32,
    learning_rate=0.001,
    dropout=0.3,
    noise=(0, 0),
    trainvalsplit=0.15,
)

staphy_configs = [
    dfrB_config,
    fusA_config,
    grlA_config,
    grlB_config,
    gyrA_config,
    ileS_config,
    pbp2_config,
    pbp4_promoter_config,
    pbp4_config,
    rpoB_config,
]
