# Seminar: Deep Learning for Molecular Biology materials - Approach with CNNs
Our approach uses a HybridGenomeNet, as in:
- one half of the network takes the One-Hot-Encoded DNA-Sequence
- other half processes an FCGR-Representation of the DNA-Sequence
In the end the results of both are combined in a linear bottleneck.
The main training file resides in src: run_training.ipynb 

# Usage Information
* For testing codes on online-GPU we sometimes used
  * [Google colab](https://colab.research.google.com/)

Although, a local machine is recommended because of the extended file-structure and easier editing of the configs.

