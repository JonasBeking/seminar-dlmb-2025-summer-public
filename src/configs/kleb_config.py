kleb_config = {
    "k" : 6,
    "pathogen" : "Klebsiella_pneumoniae_aztreonam",
    "genes" : ["acrR","gyrA","gyrB","ompK35","ompK36","ompK37","parC","rpsL"],
    "root_dir" : "../data/ds1",
    "batch_size" : 8,
    "learning_rate" : 0.001,
    "epochs" : 200,
    "class_weights" : [1.0,3.5],
    "weight_decay" : 0.01
}