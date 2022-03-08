# How to find our work ?

Most of our work on this project is contained in this file ***EA***. However, one important file is in the *train/* folder. This is the file  ***EA_train_molecules_graph_regression.py*** we made containing all the modifications in the training loop.

Here are a presentation of every files :

### GCN_comparative.ipynb

Implements a GCN on ZINC and Molecule-NET to compare the performance to the Graph Transformer results of the paper.


## visualize_attention_{filename}.ipynb files :

Those three files follow the same architecture. They are notebooks implementing the training of the Graph Transformer network on the ZINC dataset.

They all train for 100 epochs and they output every 10 epochs a random molecule of a batch with the activation scores of each nodes. This way, one can observe how the model is learning and where is the attention looking.

The scores of each node are the mean of the 8 attention scores of the heads at the end of the network (layer 10). The choice of this value as the score is a debatable and we wopuld be happy to discuss this choice with you.

### baseline

Implements the GT network as presented in the paper.

### LayerNorm

Implements the GT network using LayerNorm instead of BatchNorm.

### WLEncoding

Implements the GT network remplacing the Laplacian Positional Encoding (LapPE) with the Weisfeiler Lehman Positional Encoding (WL-PE) shortly discussed in the paper.
