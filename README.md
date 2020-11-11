# AdaptiveModeEstimation
This repository contains the official implementation of the paper Query Complexity of k-NN based Mode Estimation [1]. The Cython implementation of the subroutine Findk-NN is built up on Daniel LeJeune's [implementation](https://github.com/dlej/adaptive-knn) of [2]. The folder [RandomSampling](https://github.com/singhalanirudh18/AdaptiveModeEstimation/tree/master/RandomSampling) contains the code for the Sampling Model 1 in which we randomly sample dimensions to get distance estimate and the folder [SubGaussian](https://github.com/singhalanirudh18/AdaptiveModeEstimation/tree/master/SubGaussian) contains the code for the Sampling Model 2 in which oracle returns a noisy distance such that the noise is sub-Gaussian Random Variable.

## Data Processing
Before running the AdaptiveModeEstimation algorithm we need to process the data. First download the tiny imagenet dataset from [here](https://tiny-imagenet.herokuapp.com/) and extract it in the folder `Dataset` and after that run the file `Dataset/data_processing.py`.

## Cython Compilation
To install Cython follow instructions [here](https://cython.readthedocs.io/en/latest/src/quickstart/install.html). Before running the code you need to compile the Cython code in one of the folders using the following command :
```
$ python3 setup.py build_ext --inplace
```
After this run the file `main.py` in RandomSampling or SubGaussian to run the AdaptiveModeEstimation algorithm

[1] A. Singhal, S. Pirojiwala, N. Karamchandani, "Query Complexity of k-NN based Mode Estimation", 2020. Under review at ITW 2020. [arXiv](https://arxiv.org/abs/2010.13491)

[2] D. LeJeune, R. Heckel, R. G. Baraniuk, "Adaptive Estimation for Approximate k-Nearest-Neighbor Computations," 2019. AISTATS 2019. [arXiv](https://arxiv.org/abs/1902.09465)