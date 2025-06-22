from .config import GeneModelConfig

dfrB_config = GeneModelConfig(
    name="dfrB",
    genes=["dfrB"],
    dropout=0.7,
    epochs=200,
    weight_decay=0.1,
    noise=(0.0, 5.0),
    returnlowestvalloss=True,
)
fusA_config = GeneModelConfig(name="fusA", genes=["fusA"])
grlA_config = GeneModelConfig(name="grlA", genes=["grlA"])
grlB_config = GeneModelConfig(
    name="grlB",
    genes=["grlB"],
    learning_rate=0.0001,
    epochs=200,
    noise=(0, 0),
    dropout=0.2,
)
gyrA_config = GeneModelConfig(
    name="gyrA", genes=["gyrA"], dropout=0.7, noise=(0.0, 5.0), weight_decay=0.5,trainvalsplit=0.1
)
ileS_config = GeneModelConfig(name="ileS", genes=["ileS"])
pbp2_config = GeneModelConfig(name="pbp2", genes=["pbp2"])
pbp4_promoter_config = GeneModelConfig(name="pbp4_promoter", genes=["pbp4-promoter"])
pbp4_config = GeneModelConfig(
    name="pbp4", learning_rate=0.01, genes=["pbp4"], trainvalsplit=0.2
)
rpoB_config = GeneModelConfig(name="rpoB", genes=["rpoB"])

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
