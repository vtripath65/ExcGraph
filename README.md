![Git_logo_1](https://media.github.iu.edu/user/24867/files/e9401be6-c886-4d7d-b864-f126c168f9f8)

ExcGraph is a graph neural network to predict emission properties of dye like chromophores.
The architecture is written in "Spektral", a 
## Requirements

- Python 3.8
- spektral 1.2.0
- numpy 1.24.3
- tensorflow 2.11.1
## Deployment

To deploy this project run

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
This will generate the output file "load_saved_model.log" along with the png files in the load_model directory.