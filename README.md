# Persian Languge Modeling for COVID-2019 Corpus
This package is an implementation of persian language model. data gathered from [Lobkalam](https://lobkalam.ir) system. We place part of Data in repository. to collecting more data, please contact to [Lobkalam](https://lobkalam.ir). 

# Dependencies
This package tested on:
1. Python3.6
2. Ubuntu 20.04

# Parameters
```
--data             folder of data path
--embed_size       embedding size
--n_hidden         dimension of hidden layers
--n_layers         the number of layers
--l_rate           learning rate
--epochs           epochs number
--batch_size       batch size
--bptt             sequence length
--dropout          dropout
--save             path to save the  model
--optimzer         (SGD, Momentum, Adam)
```

# Install
`python3 -m venv env`</br>
`source env/bin/activate`</br>
`python3 -m pip install -r requirements.txt`</br>

# Usage
To training and testing model, run:</br>
`source env/bin/activate`</br>
`cd src`</br>
`python3 main.py --data ../input --embed_size 400 n_hid 400 --n_layers 2 --l_rate 20 --epochs 30 `
