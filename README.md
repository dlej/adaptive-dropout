# adaptive-dropout
Code for the paper "The Flip Side of the Reweighted Coin: Duality of Adaptive Dropout and Regularization," [arXiv:2106.07769](https://arxiv.org/abs/2106.07769).

## usage
To generate the plots for the variational dropout case study, first run [`experiments/Sparse MNIST.ipynb`](https://github.com/dlej/adaptive-dropout/blob/main/experiments/Sparse%20MNIST.ipynb). To generate both the computed effective penalties and the case study comarison plots, run [`experiments/Figures.ipynb`](https://github.com/dlej/adaptive-dropout/blob/main/experiments/Figures.ipynb).

## dependencies

The primary dependencies are `pytorch 1.8.1` with CUDA 11.1 and `skorch 0.10.1` (https://skorch.readthedocs.io/en/stable/), but we provide a complete description of the `conda` environment in [`environment.yml`](https://github.com/dlej/adaptive-dropout/blob/main/environment.yml) for completeness.
