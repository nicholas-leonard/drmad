# DrMAD

To provide an efficient and easy-to-use hyperparameter tuning toolbox for Torch deep learning ecosystems.

It combines Bayesian optimization (BO) and automatic differentiation (AD). For the Bayesian optimization module,
we will extend on [hypero](https://github.com/Element-Research/hypero); the automatic differentiation part is based on
DrMAD method, https://arxiv.org/abs/1601.00917.

It is the only tool that can tune thousands of continuous hyperparameters (e.g. L2 penalties for each neuron or
learning rates for each layer) with a reasonable time/computational budget -- reads: outside Google.

## Current Status
Skechy code for tuning L2 penalties and learning rates on MNIST dataset with CUDA support.

## TODO
1. API for tuning learning rates, weight decay and momentum. 
2. Experiments on ImageNet


## Dependencies
* [Twitter Torch Autograd](https://github.com/twitter/torch-autograd): the next version will not depend on this. 

## How to run

- `drmad_mnist.lua` is for tuning L2 penalties on MNIST. 
- `cuda_drmad_mnist.lua` is for tuning L2 penalties on MNIST with CUDA. 
- `lr_drmad_mnist.lua` is for tuning learning rates and L2 penalties on MNIST.  

## Tricks

### Rally with ([Net2Net](https://github.com/soumith/net2net.torch))
ImageNet dataset usually needs ~450,000 iterations. DrMAD may not approxiate this long trajectory well. 

One approach would be to repeatedly initialize the weights using Net2Net, from small subsets to larget subsets
and finally to the full dataset.

### Acceleration with Model Compression
We will add a regression loss at every layer, which is also used in [Deep Q-Networks for Accelerating the Training of Deep Neural Networks](https://arxiv.org/abs/1606.01467). However, the aim here is not to compress the model, so we do not decrease the number of parameters. 

### BO and AD
BO is a global optimization method (it can handle 20 hyperparameters at most), whereas AD can only find local solutions
(it can handle thousands of hyperparameters because it uses gradient information). We first use BO to get some initial
average hyperparameters. Then we use AD method to further search for diverse local hyperparameters.

## Contact

If you have any problems or suggestions, please contact me: jie.fu A~_~T u.nus.edu~~cation~~