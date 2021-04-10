# PyTorchTutorial
This repository contains my snippets and sample codes for developing deep learning application with Pytorch.

## Installation
Create environment:

`conda create -n PythonTutorial`

Activate it:

`conda activate PythonTutorial`

Install pytorch, torchvision and cuda:

`conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch`

Install tensorboard:

`conda install  tensorboard`

Install matplotlib:

`conda install -c conda-forge matplotlib`

Install `torchviz` for visualizations of execution graphs 

`pip install torchviz`

and 

`conda install  pydot`

If you want to view the <b>dot</b> file install `xdot`

`sudo apt-get install xdot`

Install jupyter:

`conda install  jupyter`


To updated all packages:

`conda update -n PythonTutorial  --all`

## Tutorials

[Activation Functions](src/activation_functions/activation_function.ipynb)
[Convolution](src/conv/cross_correlation_convolution.ipynb)
- [Cross Correlation](conv/cross_correlation_convolution.ipynb#Cross-Correlation)
- [Convolution](conv/cross_correlation_convolution.ipynb#Convoloution)
- [Shape of the Convolution Output](conv/cross_correlation_convolution.ipynb#Shape-of-the-Convolution-Output)
- [2D Convolution as Matrix Multiplication](conv/cross_correlation_convolution.ipynb#2D-Convolution-as-Matrix-Multiplication)
- [Convolution in RGB Images](conv/cross_correlation_convolution.ipynb#Convolution-in-RGB-Images)
- [Transpose Convolution](conv/cross_correlation_convolution.ipynb#Transpose-Convolution)
- [1x1 Convolution (Network-in-Network)](cross_correlation_convolution.ipynb#1x1-Convolution:-Network-in-Network)
- [Dilated Convolutions](conv/cross_correlation_convolution.ipynb#Dilated-Convolutions)

