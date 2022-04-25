# Batch size reconstruction-distribution trade-off in kernel based generative autoencoders

## Repository info

This repository contains an implementation of `Batch size reconstruction-distribution trade-off in kernel based generative autoencoders` in PyTorch, proposed by Szymon Knop, Przemysław Spurek, Marcin Mazur, Jacek Tabor and Igor Podolak.

## Contents of the repository

```text
|-- src/ - contains an implementation of the models proposed in the paper allowing to reproduce experiments from the original paper
|---- architecture/ - files containing architectures proposed in the paper
|---- factories/ - factories used to create objects proper objects base on command line 
arguments. Subfolders contain factories for specific models
|---- lightning_callbacks/ - implementation of evaluators of metrics reported in our experiments
|---- lighting_modules/ - implementation of experiments in pytorch lightning
|---- metrics/ - directory containing the implementation of all of the metrics used in paper
|---- modules/ - custom neural network layers used in models
|---- train_autoencoder.py - the main script to run all of the experiments
|-- results/ - directory that will be created to store the results of conducted experiments
|-- data/ - default directory that will be used as a source of data and place to download datasets
```

Experiments are written in `pytorch-lightning` to decouple the science code from the engineering. The `LightningModule` implementation is in `train_autoencoder.py` file. For more details refer to [PyTorch-Lightning documentation](https://github.com/PyTorchLightning/pytorch-lightning)

## Conducting the experiments

Below there is simple script that runs the experiment:

`python -m train_autoencoder --model wae-mmd --dataset fmnist --latent_dim 8 --batch_size 16 --memory_length 112 --gpus 1 --check_val_every_n_epoch 1 --max_epochs 500 --lambda_val 64 --log_generative`

## Browsing the results

Results are stored in tensorboard format. To browse them run the following command:
`tensorboard --logdir results`

## Other options

The code allows manipulating some of the parameters(for example using other versions of the model, changing learning rate values) for more info see the list of available arguments in `src/args_parser.py` file

To run the unit tests execute the following command:
`python -m unittest`

# Datasets

The repository uses default datasets provided by PyTorch for MNIST, FashionMNIST, CIFAR-10 and CELEBA. To convert CELEB-A to 64x64 images we first center crop images to 140x140 and then resize them to 64x64.

# Environment

- python3
- pytorch
- torchvision
- numpy
- pytorch-lightning

# Acknowledgments

All of the implementations are based on the respective papers and repositories.

- For Wasserstein AutoEncoders [arXiv](https://arxiv.org/abs/1711.01558) and [GitHub repository](https://github.com/tolstikhin/wae)

- For Cramer-Wold AutoEncoders [JMLR](https://jmlr.org/papers/v21/19-560.html) and [GitHub repository](https://github.com/gmum/cwae-pytorch)

The research of S. Knop was funded by the Priority Research Area Digiworld under the program Excellence Initiative - Research University at the Jagiellonian University in Kraków. 

# License

This implementation is licensed under the MIT License
