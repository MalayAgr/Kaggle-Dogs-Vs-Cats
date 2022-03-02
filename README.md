# <!-- omit in toc --> Dogs vs. Cats - Kaggle

A complete pipeline for training and tuning (using [Optuna](https://optuna.org/)) a convolutional neural network to classify cats and dogs. The pipeline contains many ideas that can be reused in multiple projects.

**Competition link**: [Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats).

**Notebook link**: [Dogs vs. Cats vs. PyTorch - Training a CNN Model](https://www.kaggle.com/defcodeking/dogs-vs-cats-vs-pytorch-training-a-cnn-mode).


## <!-- omit in toc --> Table of Contents

- [Requirements](#requirements)
- [What's included](#whats-included)
- [Model](#model)
- [Optuna Tuning](#optuna-tuning)
- [Data](#data)
- [Run](#run)

## Requirements

- Python &ge; 3.8
- PyTorch
- Scikit-learn
- Albumentations
- Optuna
- NumPy
- Pandas
- Pillow
- Rich

## What's included

- [config.py](src/config.py) - Contains static parameters (parameters that are not tuned) like number of epochs and directories.
- [dataset.py](src/dataset.py)- Implements a PyTorch Dataset to load the train and test data.
- [training.py](src/training.py) - Implements an Engine class to encapsulate training of a model using some optimizer and optional learning rate scheduler.
- [model.py](src/model.py) - Implements a PyTorch model that can be built entirely from parameters supplied by Optuna (number of output channels of convolutional layers, kernel sizes, number of output features of linear layers, dropout).
- [main.py](src/main.py) - Main driver for the pipeline.
- [console.py](src/console.py) - Global `rich` console used for pretty output.

## Model

By default, the model has the following architecture:

- Four convolutional blocks. Each block has the following components:
  - `nn.Conv2d()` with no bias
  - `nn.BatchNorm2d()`
  - `nn.ReLU()`
  - `nn.MaxPool2d()` with a pool size of `(2, 2)`
- `nn.Flatten()`
- `nn.Dropout()`
- Three linear blocks. Each block has the following components:
  - `nn.Linear()`
  - `nn.ReLU()`
- Output layer `nn.Linear()` with one output node
- `nn.Sigmoid()`

## Optuna Tuning

The following parameters are tuned using Optuna:

- Number of output channels of each of the four `nn.Conv2d()` layers.
- Kernel sizes but only two unique values are tuned. The first two and next two `nn.Conv2d()` layers have the same kernel size.
- Dropout probability for the `nn.Dropout()` layer.
- Number of output features of each of the three `nn.Linear()` layers.
- Learning rate
- Batch size

## Data

- Create a directory which matches `DATA_DIR` in [config.py](src/config.py)

```shell
mkdir <DATA_DIR>
```

- Download the dataset using the `kaggle` API ([guide](https://github.com/Kaggle/kaggle-api)) or from the competition page inside `DATA_DIR`:

```shell
kaggle competitions download -c dogs-vs-cats
```

- Unzip `train.zip` and `test1.zip` inside `DATA_DIR`:

```shell
unzip <DATA_DIR>/train.zip -d <DATA_DIR> && unzip <DATA_DIR>/test1.zip -d <DATA_DIR>
```

## Run

- Clone the repository:

```shell
git clone https://github.com/MalayAgr/Kaggle-Dogs-Vs-Cats.git
```

- `cd` into the repository:

```shell
cd Kaggle-Dogs-Vs-Cats
```

- Execute the `main.py` script:

```shell
python src/main.py
```
