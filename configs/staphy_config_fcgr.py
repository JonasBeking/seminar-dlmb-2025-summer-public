from .config import GeneModelConfig

dfrB_config = GeneModelConfig(
    name="dfrB",
    genes=["dfrB"],
    dropout=0.05,
    epochs=200,
    weight_decay=0.01,
    noise=(0.0, 1.0),
    learning_rate=0.0001,
    batch_size=16,
    trainvalsplit=0.15,
    onlyfcgr=True,
)
fusA_config = GeneModelConfig(
    name="fusA",
    genes=["fusA"],
    dropout=0.1,
    onlyfcgr=True,
    noise=(0, 0),
    weight_decay=0.0,
    batch_size=32
)
grlA_config = GeneModelConfig(
    learning_rate=0.0001,
    name="grlA",
    genes=["grlA"],
    dropout=0.02,
    batch_size=16,
    weight_decay=0.2,
    noise=(0.0, 0.25),
    onlyfcgr=True,
)
grlB_config = GeneModelConfig(
    name="grlB",
    genes=["grlB"],
    learning_rate=0.0001,
    batch_size=32,
    epochs=200,
    noise=(0, 0.0),
    dropout=0.1,
    weight_decay=0.05,
    trainvalsplit=0.2,
    onlyfcgr=True,
)
gyrA_config = GeneModelConfig(
    name="gyrA",
    genes=["gyrA"],
    batch_size=16,
    learning_rate=0.0001,
    dropout=0.15,
    epochs=200,
    noise=(0.0, 0.5),
    weight_decay=0.05,
    trainvalsplit=0.15,
    onlyfcgr=True,
)
ileS_config = GeneModelConfig(
    k=6,
    name="ileS",
    genes=["ileS"],
    trainvalsplit=0.15,
    dropout=0.06,
    noise=(0.0, 0.0),
    weight_decay=0.1,
    batch_size=64,
    epochs=200,
    learning_rate=0.0001,
    onlyfcgr=True,
)
pbp2_config = GeneModelConfig(
    name="pbp2",
    genes=["pbp2"],
    dropout=0.15,
    batch_size=64,
    trainvalsplit=0.25,
    onlyfcgr=True,
    learning_rate=0.0001,
)
pbp4_promoter_config = GeneModelConfig(
    name="pbp4_promoter", genes=["pbp4-promoter"], dropout=0.1, onlyfcgr=True
)
pbp4_config = GeneModelConfig(
    batch_size=108,
    rareclasssampling=False,
    name="pbp4",
    learning_rate=0.01,
    genes=["pbp4"],
    dropout=0.3,
    trainvalsplit=0.2,
    returnlowestvalloss=False,
    rareclasssamplerreplacement=False,
    lossweighting=True,
    onlyfcgr=True,
)
rpoB_config = GeneModelConfig(name="rpoB", genes=["rpoB"], dropout=0.1, onlyfcgr=True)

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
