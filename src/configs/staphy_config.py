from .config import GeneModelConfig

dfrB_config = GeneModelConfig(genes=["dfrB"])
fusA_config = GeneModelConfig(genes=["fusA"])
grlA_config = GeneModelConfig(genes=["grlA"])
grlB_config = GeneModelConfig(genes=["grlB"])
gyrA_config = GeneModelConfig(genes=["gyrA"])
ileS_config = GeneModelConfig(genes=["ileS"])
pbp2_config = GeneModelConfig(genes=["pbp2"])
pbp4_promoter_config = GeneModelConfig(genes=["pbp4-promoter"])
pbp4_config = GeneModelConfig(genes=["pbp4"],epochs=20)
rpoB_config = GeneModelConfig(genes=["rpoB"])

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
