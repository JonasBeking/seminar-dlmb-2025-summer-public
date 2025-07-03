from .config import GeneModelConfig

dfrB_config = GeneModelConfig(
    name="dfrB",
    genes=["dfrB"],
    epochs=200,
    noise=(0, 2.0),
    dropout=0.1,
    trainvalsplit=0.2,
    batch_size=64,
    learning_rate=0.001,
    onlysequence=True
)
fusA_config = GeneModelConfig(
    name="fusA",
    genes=["fusA"],
    epochs=200,
    noise=(0, 2.0),
    dropout=0.1,
    trainvalsplit=0.2,
    batch_size=64,
    learning_rate=0.001,
    onlysequence=True
)
grlA_config = GeneModelConfig(
    name="grlA",
    genes=["grlA"],
    epochs=200,
    noise=(0, 2.0),
    dropout=0.1,
    trainvalsplit=0.2,
    batch_size=64,
    learning_rate=0.001,
    onlysequence=True
)
grlB_config = GeneModelConfig(
    name="grlB",
    genes=["grlB"],
    epochs=200,
    noise=(0, 2.0),
    dropout=0.1,
    trainvalsplit=0.2,
    batch_size=64,
    learning_rate=0.001,
    onlysequence=True
)
gyrA_config = GeneModelConfig(
    name="gyrA",
    genes=["gyrA"],
    epochs=200,
    noise=(0, 2.0),
    dropout=0.1,
    trainvalsplit=0.2,
    batch_size=64,
    learning_rate=0.001,
    onlysequence=True
)
ileS_config = GeneModelConfig(
    name="ileS",
    genes=["ileS"],
    epochs=200,
    noise=(0, 2.0),
    dropout=0.1,
    trainvalsplit=0.2,
    batch_size=64,
    learning_rate=0.001,
    onlysequence=True
)
pbp2_config = GeneModelConfig(
    name="pbp2",
    genes=["pbp2"],
    epochs=200,
    noise=(0, 2.0),
    dropout=0.1,
    trainvalsplit=0.2,
    batch_size=64,
    learning_rate=0.001,
    onlysequence=True
)
pbp4_promoter_config = GeneModelConfig(
    name="pbp4_promoter",
    genes=["pbp4-promoter"],
    epochs=200,
    noise=(0, 2.0),
    dropout=0.1,
    trainvalsplit=0.2,
    batch_size=64,
    learning_rate=0.001,
    onlysequence=True
)
pbp4_config = GeneModelConfig(
    name="pbp4",
    genes=["pbp4"],
    epochs=200,
    noise=(0, 2.0),
    dropout=0.1,
    trainvalsplit=0.2,
    batch_size=64,
    learning_rate=0.001,
    onlysequence=True
)
rpoB_config = GeneModelConfig(
    name="rpoB",
    genes=["rpoB"],
    epochs=200,
    noise=(0, 2.0),
    dropout=0.1,
    trainvalsplit=0.2,
    batch_size=64,
    learning_rate=0.001,
    onlysequence=True
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
