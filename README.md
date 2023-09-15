![Git_logo_1](https://media.github.iu.edu/user/24867/files/e9401be6-c886-4d7d-b864-f126c168f9f8)

ExcGraph is a graph neural network to predict emission properties of dye like chromophores.
The architecture is written in "Spektral", a 
## Requirements

- Python 3.10.10
- spektral 1.2.0
- matplotlib 3.7.1
- numpy 1.24.3
- tensorflow 2.11.1
- pip 23.1.2
## Deployment

To deploy this project run

```bash
  git clone https://github.iu.edu/vtripath/ExcGraph.git
  cd ExcGraph
  make conda_env
  conda activate ExcGraph
  cd data
  unzip pickled_data.zip
```
Note: This is an unpublished work. Thus, I have not provided the data here. If you really want to run this code please ask me for the data.

If you have the data you will have to run the following code:

```bash
  cd data
  unzip pickled_data.zip
```

To train the model use
```bash
  cd ../training
  python spektral-model.py > spektral-model.log
```
This will generate the output file "spektral-model.log" along with the png files in the training directory.

To use the pretrained model on external dataset use the following:
```bash
  cd ../load_model
  python load_saved_model.py > load_saved_model.log
```
This will generate the output file "load_saved_model.log" along with the png files in the load_model directory. The purpose of this directory is to test the model on new external test sets.
## Architeture
![git_architecture](https://media.github.iu.edu/user/24867/files/5acdadcc-45a0-4b6c-9aa4-53fb47289279)
Two convolution layers followed by a pooling layers to obtain molecular fingerprint. This fingerprint is passed through a feedforward neural network to obtain the predicted emission maxima. The node features contain atomic properties and bonding environment along with the eps value (dielectric) of the solvent. The solvent index will be the same for all the nodes for a chromophore in a particular solvent. Using this as a global property has been tested with no improvement.
