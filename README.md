# PyTorchTutorial
This repository contains my snippets and sample codes for developing deep learning application with Pytorch.

## Installation
1. Create environment:

```
conda create -n PyTorchTutorial
```

2. Activate it:

```
conda activate PyTorchTutorial
```

3. Install pytorch, torchvision, cuda tensorboard, jupyter, matplotlib, pydot:

```
conda install pytorch torchvision  pytorch-cuda -c pytorch -c nvidia 
conda install tensorboard
conda install -c conda-forge matplotlib  
conda install pydot
conda install -c conda-forge jupyterlab
conda install anaconda::scikit-learn
conda install conda-forge::seaborn
```


4. Install `torchviz` for visualizations of execution graphs 

```
pip install torchviz
```

If you want to view the <b>dot</b> file install `xdot`

```
sudo apt-get install graphviz
sudo apt-get install xdot
```

5. To updated all packages:

```
conda update -n PyTorchTutorial  --all
```


6. set up the soft-link to repo:
```
cd /home/$USER/workspace/
git clone git@github.com:behnamasadi/PyTorchTutorial.git
ln -s /home/$USER/workspace/PyTorchTutorial /home/$USER/anaconda3/envs/PyTorchTutorial/src
```



## Tutorials

[Real World Practices for Training and Regularization](PyTorch_training_template/index.ipynb)


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

  
[Serialization, Saving and Loading Tensors/ Networks](serialization_saving_loading/index.ipynb)  
[Tensorboard](tensorboard/index.ipynb)  
[Visualization of Graph](torchviz_visualize_graphs/index.ipynb)


## [Neural Network Basics](#)
- [Activation Functions](activation_functions/activation_function.ipynb)  
- [Loss Functions](loss_functions/loss_functions.ipynb)  


## [Training Process](#)

- [Optimizer Package](optim_package/index.pynb)  
- [Learning Rate & Schedulers (Step, Cosine Annealing, etc.)](optim_package/index.ipynb#Learning-Rate-Schedulers-(Schedulers))
- Overfitting & Underfitting
- [Regularization](regularization/index.ipynb)  
- [Batch Normalization](batch_normalization/index.ipynb)
- [Layer Normalization](layer_normalization/index.ipynb)
- Weight Initialization Strategies
- Evaluation vs Training Mode
- [Real World Practices and PyTorch training template](Real World Practices and PyTorch training template)
- [Drop out](drop_out/index.ipynb)  
---

## [**PyTorch Fundamentals**](#) 
- PyTorch Tensor Basics & Data Types
- Autograd and Computational Graph
  - `requires_grad`, `no_grad`, `detach()`, `zero_grad()`
- `nn.Module` and Custom Models
- Model Saving & Loading (Serialization)
- TensorBoard & Model Debugging
- Visualizing Model Graphs & Gradients

---


## [**CNNs and Visual Learning**](#) 
Vision-focused models and operations.

### [CNN Building Blocks](#)
- [Convolution vs Cross-Correlation](conv/cross_correlation_convolution.ipynb#1.-Cross-Correlation)
- [Shape of Output](cross_correlation_convolution.ipynb#4.Shape-of-the-Convolution-Output)
- [RGB Image Convolution](conv/cross_correlation_convolution.ipynb#5.Convolution-in-RGB-Images)
- [Convolution as Matrix Multiplication](conv/cross_correlation_convolution.ipynb#Convolution-as-Matrix-Multiplication)
- [Conv2d class vs conv2d function](conv/cross_correlation_convolution.ipynb#PyTorch-Conv2d-class-vs-conv2d-function)
- [Unfold/ fold](cross_correlation_convolution.ipynb#torch.nn.Unfold)
- [Padding, Stride, Dilation](conv/cross_correlation_convolution.ipynb#4.Shape-of-the-Convolution-Output)
- [Pooling (Max, Average, Adaptive)](conv/cross_correlation_convolution.ipynb#Pooling)
- [1x1, Dilated, Transposed Convolution (Upsampling) Convolution](conv/cross_correlation_convolution.ipynb#3.-Most-Common-Types-of-Convolution-in-Deep-Learning)
- [Feature Map](conv/cross_correlation_convolution.ipynb#8.-Feature-Map)

### [Modern CNN Architectures](#)
- LeNet, AlexNet, VGG
- ResNet (Skip Connections)
- EfficientNet, MobileNet (for mobile)

### [Image Preprocessing & Augmentation Workflows](#)
- [Dataset and DataLoader APIs](data_loader_pre_processing/datasets_loader.ipynb)  
- [Pre Processing Transforms](data_loader_pre_processing/index.ipynb#1.-Preprocessing-Transforms)  
- [Data Augmentation Transforms](data_loader_pre_processing/index.ipynb#2.-Data-Augmentation-Transforms)  
- [Training/Evaluation Transform Pipeline](data_loader_pre_processing/index.ipynb#3.-Building-the-Transform-Pipeline)  

  

---

- [Transfer learning](transfer_learning/transfer_learning.ipynb)  


## [**Sequence Modeling**](#)
- Encoder–Decoder Architecture
- Teacher Forcing


---

## [**Attention & Transformers**](#) 
This is where modern DL models start to shine.

- Attention Mechanism (Additive, Dot-Product)
- Self-Attention & Multi-Head Attention
- Positional Encoding
- Transformer Encoder/Decoder
- Vision Transformer (ViT)
- Transformer Applications (BERT, GPT overview)

---

## [**Advanced Topics & Research Trends**](#) 
Optional but valuable for deeper exploration or research.

- Autoencoders (Vanilla, Variational)
- GANs (Generator, Discriminator, Losses)
- Diffusion Models (Denoising Score Matching)
- Contrastive Learning (SimCLR, MoCo)
- Self-Supervised Learning
- Zero-shot & Few-shot Learning
- Large Language Models (LLMs)

---

## [**Practical Engineering & Utilities**](#) 

- Experiment Tracking (TensorBoard, WandB)
- Data Versioning
- Model Deployment (ONNX, TorchScript)
- Quantization & Pruning
- Inference Optimization
- Logging and Debuggingv
- Project Structure & Best Practices

---


