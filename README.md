![image](https://github.com/user-attachments/assets/b3301c0f-e3ac-4802-9338-7d9a476829ae)


ExcGraph is a graph neural network to predict emission properties of dye like chromophores.
The architecture is written in "Spektral", a Python library for graph deep learning, based on the Keras API and TensorFlow.
## Requirements

- Python 3.10.10
- spektral 1.2.0
- matplotlib 3.7.1
- numpy 1.24.3
- tensorflow 2.11.1
- pip 23.1.2

Note: You do not have to install them yourselves. Following the proceedure in the deployment section will get your environment set up for you.

## Deployment

To deploy this project run

```bash
  git clone https://github.com/vtripath65/ExcGraph.git
  cd ExcGraph
  make conda_env
  conda activate ExcGraph
```
Note: I have not provided the data here. If you really want to run this code please contact me for the data.

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
## Architecture
![image](https://github.com/user-attachments/assets/130cc4e8-3882-4f6d-8a64-508bd5207141)


Two convolution layers followed by a pooling layers to obtain molecular fingerprint. This fingerprint is passed through a feedforward neural network to obtain the predicted emission maxima. The node features contain atomic properties and bonding environment along with the eps value (dielectric) of the solvent. The solvent index will be the same for all the nodes for a chromophore in a particular solvent. Using this as a global property has been tested with no impro
vement.
## Example Result
![image](https://github.com/user-attachments/assets/127146da-eae7-4c9c-911d-e64ec5c66bbf)


The predictions are colored using different dyes classes. The chromophores are preprocessed to assign each chromophore to its own corresponding dye class.
## Differentiating features

- Test on external dataset
- Shows good extrapolation performance
- Unlike previous work, no constraint on the size of the chromophores
- State of the art performance with test error being ~0.1 eV
