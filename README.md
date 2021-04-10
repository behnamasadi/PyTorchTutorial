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

[Activation Functions](src/activation_functions/activation_function.ipynb)
[Convolution](src/conv/cross_correlation_convolution.ipynb)
- [Cross Correlation](conv/cross_correlation_convolution.ipynb#Cross-Correlation)
- [Convolution](conv/cross_correlation_convolution.ipynb#Convoloution)
- [Shape of the Convolution Output](conv/cross_correlation_convolution.ipynb#Shape-of-the-Convolution-Output)
- [2D Convolution as Matrix Multiplication](conv/cross_correlation_convolution.ipynb#2D-Convolution-as-Matrix-Multiplication)
- [Convolution in RGB Images](conv/cross_correlation_convolution.ipynb#Convolution-in-RGB-Images)
- [Transpose Convolution](conv/cross_correlation_convolution.ipynb#Transpose-Convolution)
- [1x1 Convolution Network-in-Network](cross_correlation_convolution.ipynb#1x1-Convolution:-Network-in-Network)
- [Dilated Convolutions](conv/cross_correlation_convolution.ipynb#Dilated-Convolutions)

[Data Loader/ Pre Processing](src/data_loader_pre_processing/index.ipynb)
[Pytorch Data Types](data_types/index.ipynb)
[Encoder/ Decoder](src/encoder/index.ipynb)
[Grad Package](src/grad_package)
- [Autograd]()
- [Dynamic Computational Graph]()
- [Detach]()
- [Fine-tuning]()
- [Leaf Tensor]()
- [No Grad]()
- [Zero Grad]()

[Image Captioning](src/image_captioning)
[Learning Steps Strategy]()

[Loss Functions](loss_functions/loss_functions.ipynb)
[LSTM](src/LSTM/index.ipynb)
[Optimizer Package](src/optim_package/index.pynb)
[Pooling](src/pooling/index.ipynb)
- [Max Pool](pooling/index.ipynb#Max-Pool)
- [Average Pool](pooling/index.ipynb#Average-Pool)
- [Adaptive Average Pool](pooling/index.ipynb#Adaptive-Average-Pool)

[RNN](src/rnn/index.ipynb)
[Serialization](src/serialization/index.ipynb)
[Tensorboard](src/tensorboard/index.ipynb)
[Transfer learning](src/transfer_learning/transfer_learning.ipynb)
[Visualization of Graph](src/graph_visualization/index.ipynb)













