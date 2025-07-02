from .config import GeneModelConfig

dfrB_config = GeneModelConfig(
    name="dfrB",
    genes=["dfrB"],
    dropout=0.1,
    epochs=300,
    weight_decay=0.01,
    noise=(0.0, 2.0),
    returnlowestvalloss=False,
)
fusA_config = GeneModelConfig(name="fusA", genes=["fusA"], dropout=0.1,epochs=300,)
grlA_config = GeneModelConfig(
    name="grlA",
    genes=["grlA"],
    dropout=0.1,
    noise=(0.0, 5.0),
    epochs=300,
)
grlB_config = GeneModelConfig(
    name="grlB",
    genes=["grlB"],
    learning_rate=0.001,
    noise=(0, 2.0),
    dropout=0.25,
    trainvalsplit=0.2,
    batch_size=16,
    epochs=300
)
gyrA_config = GeneModelConfig(
    name="gyrA",
    genes=["gyrA"],
    dropout=0.1,
    noise=(0.0, 0.0),
    weight_decay=0.2,
    trainvalsplit=0.2,
    epochs=300
)
ileS_config = GeneModelConfig(
    k=6,
    name="ileS",
    genes=["ileS"],
    trainvalsplit=0.2,
    dropout=0.1,
    noise=(0.0, 0.0),
    batch_size=32,
    learning_rate=0.0001,
    epochs=300,
)
pbp2_config = GeneModelConfig(name="pbp2", genes=["pbp2"],dropout=0.2,batch_size=64,trainvalsplit=0.25,epochs=300,learning_rate=0.001)
pbp4_promoter_config = GeneModelConfig(name="pbp4_promoter", genes=["pbp4-promoter"],dropout=0.1,epochs=300)
pbp4_config = GeneModelConfig(
    batch_size=108,
    rareclasssampling=True,
    name="pbp4",
    learning_rate=0.01,
    genes=["pbp4"],
    dropout=0.3,
    trainvalsplit=0.2,
    returnlowestvalloss=False,
    rareclasssamplerreplacement=False,
    lossweighting=True,
    epochs=300
)
rpoB_config = GeneModelConfig(name="rpoB", genes=["rpoB"],dropout=0.1,epochs=300)

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
