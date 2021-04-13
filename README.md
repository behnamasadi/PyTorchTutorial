# PyTorchTutorial
This repository contains my snippets and sample codes for developing deep learning application with Pytorch.

## Installation
1. Create environment:

`conda create -n PythonTutorial`

2. Activate it:

`conda activate PythonTutorial`

3. Install pytorch, torchvision, cuda tensorboard, jupyter, matplotlib, pydot:

`conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch`  
`conda install tensorboard`  
`conda install -c conda-forge matplotlib`  
`conda install pydot`  
`conda install jupyter`  

4. Install `torchviz` for visualizations of execution graphs 

`pip install torchviz`

If you want to view the <b>dot</b> file install `xdot`

`sudo apt-get install xdot`

5. To updated all packages:

`conda update -n PythonTutorial  --all`

## Tutorials

[Activation Functions](activation_functions/activation_function.ipynb)  
[Convolution](conv/cross_correlation_convolution.ipynb)  
- [Cross Correlation](conv/cross_correlation_convolution.ipynb#Cross-Correlation)  
- [Convolution](conv/cross_correlation_convolution.ipynb#Convoloution)  
- [Shape of the Convolution Output](conv/cross_correlation_convolution.ipynb#Shape-of-the-Convolution-Output)  
- [2D Convolution as Matrix Multiplication](conv/cross_correlation_convolution.ipynb#2D-Convolution-as-Matrix-Multiplication)  
- [Convolution in RGB Images](conv/cross_correlation_convolution.ipynb#Convolution-in-RGB-Images)  
- [Transpose Convolution](conv/cross_correlation_convolution.ipynb#Transpose-Convolution)  
- [1x1 Convolution Network-in-Network](conv/cross_correlation_convolution.ipynb#1x1-Convolution-Network-in-Network)  
- [Dilated Convolutions](conv/cross_correlation_convolution.ipynb#Dilated-Convolutions)  

[Data Loader/ Pre Processing](data_loader_pre_processing/index.ipynb)  
[Pytorch Tensor Data Types](data_types/index.ipynb)  
[Encoder/ Decoder](encoder/index.ipynb)  
[Grad Package](grad_package/)  
- [Computational Graph](grad_package/grad.ipynb#Computational-Graph)
- [Autograd](grad_package/grad.ipynb#Autograd)  
- [Dynamic Computational Graph](grad_package/grad.ipynb#)  
- [Detach](grad_package/grad.ipynb#detach)  
- [Exclusion from the DAG](grad_package/grad.ipynb#Exclusion-from-the-DAG)  
- [Leaf Tensor](grad_package/grad.ipynb#Leaf)  
- [No Grad](grad_package/grad.ipynb#no_grad())  
- [Zero Grad](grad_package/grad.ipynb#zero_grad)  

[Image Captioning](image_captioning)  
[Learning Steps Strategy](learning_steps_strategy/index.ipynb#Pre-process-Data)  
- [Pre-process Data](learning_steps_strategy/index.ipynb#Pre-process-Data)  
- [Choose Architecture](learning_steps_strategy/index.ipynb#Choose-Architecture)  
- [Weight initialization](learning_steps_strategy/index.ipynb#Weight-initialization)  
- [Batch Normalization](learning_steps_strategy/index.ipynb#Batch-Normalization)  
- [Drop out](learning_steps_strategy/index.ipynb#Drop-out)  
- [Eval](learning_steps_strategy/index.ipynb#Eval)  
- [Training Steps](learning_steps_strategy/index.ipynb#Training-Steps)  
- [General Tips](learning_steps_strategy/index.ipynb#General-Tips)  

[Loss Functions](loss_functions/loss_functions.ipynb)  
[LSTM](LSTM/index.ipynb)  
[Optimizer Package](optim_package/index.pynb)  
[Pooling](pooling/index.ipynb)  
- [Max Pool](pooling/index.ipynb#Max-Pool)  
- [Average Pool](pooling/index.ipynb#Average-Pool)  
- [Adaptive Average Pool](pooling/index.ipynb#Adaptive-Average-Pool)  

[RNN](rnn/index.ipynb)  
[Serialization, Saving and Loading Tensors/ Networks](serialization_saving_loading/index.ipynb)  
[Tensorboard](tensorboard/index.ipynb)  
[Transfer learning](transfer_learning/transfer_learning.ipynb)  
[Visualization of Graph](graph_visualization/index.ipynb)  
