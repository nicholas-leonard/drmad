# DrMAD

To provide an efficnet and easy-to-use hyperparameter tuning toolbox for Torch deep learning ecosystems. 

It combines Bayesian optimization (BO) and automatic differentiation (AD). For the Bayesian optimization module, we will extend on [hypero](https://github.com/Element-Research/hypero); the automatic differentiation part is based on DrMAD method, https://arxiv.org/abs/1601.00917. 

It will be the only tool that can tune thousands of continuous hyperparameters (e.g. L2 penalties for each neuron or learning rates for each layer) with a reasonable time/computational budget -- reads: outside Google. 

## Current Status
Only skechy code for L2 penalties on MNIST dataset. 

## TODO
1. Experiments on CIFAR-10 and ImageNet
2. Support for learning rates
3. Refactoring


## Dependencies
* Twitter Torch Autograd: https://github.com/twitter/torch-autograd

## Tricks

### Rally with ([Net2Net](https://github.com/soumith/net2net.torch))
ImageNet dataset usually needs ~450,000 iterations. DrMAD may not approxiate this long trajectory well. 

One approach would be to repeatedly initialize the weights using Net2Net, from small subsets to larget subsets and finally to the full dataset. 

### BO and AD
BO is a global optimization method, whereas AD can only find local solutions. We first use BO to get some initial average hyperparameters (10 at most). Then we use AD method to further search for diverse local hyperparameters. 

