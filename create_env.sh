#!/usr/bin/bash

echo 'Creating conda environment for ExcGraph'
conda env create -q -f devtools/environment.yml --force
export PYTHONPATH="$PYTHONPATH:$PWD"

echo 'To activate this environment, please use'
echo '#'
echo '#      $ conda activate ExcGraph'
echo '#'
