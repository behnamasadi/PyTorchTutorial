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


4. Install `torchviz` for visualizations of execution graphs and `mlflow` and `wandb` for experiment tracking  

```
pip install torchviz
pip install mlflow
pip install wandb
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


## [**PyTorch Fundamentals**](#) 
- [PyTorch Tensor Basics & Data Types](data_types/index.ipynb)  
- [Grad Package](grad_package/)  
    - [Computational Graph](grad_package/grad.ipynb#Computational-Graph)  
    - [Autograd](grad_package/grad.ipynb#Autograd)  
    - [Dynamic Computational Graph](grad_package/grad.ipynb#)  
    - [Detach](grad_package/grad.ipynb#detach)  
    - [Exclusion from the DAG](grad_package/grad.ipynb#Exclusion-from-the-DAG)  
    - [Leaf Tensor](grad_package/grad.ipynb#Leaf)  
    - [No Grad](grad_package/grad.ipynb#no_grad())  
    - [Zero Grad](grad_package/grad.ipynb#zero_grad)  
    - [Model Saving & Loading (Serialization)](serialization_saving_loading/index.ipynb)  

## [Neural Network Basics](#)
- [Neural Networks, Manifolds, and Topology](https://colah.github.io/posts/2014-03-NN-Manifolds-Topology/)  
- [Back Propagation](backpropagation/index.ipynb)  
- [Activation Functions](activation_functions/index.ipynb)  
- [Loss Functions](loss_functions/index.ipynb)  


## [Training Process](#)

- [Optimizer Package](optim_package/index.ipynb)  
- [Learning Rate & Schedulers (Step, Cosine Annealing, etc.)](optim_package/index.ipynb#Learning-Rate-Schedulers-(Schedulers))
- Overfitting & Underfitting
- [Regularization](regularization/index.ipynb) 
- [Dropout Layers](dropout_layers/index.ipynb)
- [Batch Normalization](batch_normalization/index.ipynb)
- [Layer Normalization](layer_normalization/index.ipynb)
- [Weight Initialization Strategies](weight_initialization/index.ipynb)
- [Evaluation vs Training Mode](learning_monitoring/index.ipynb)
    * [Training, Validation, and Test Set](learning_monitoring/index.ipynb#Training-and-Validation-set)
    * [Monitor for Overfitting](learning_monitoring/index.ipynb#1.-Monitor-for-Overfitting)
    * [Early Stopping](index.ipynb#2.-Implement-Early-Stopping)
    * [Visualize Metrics](learning_monitoring/index.ipynb#4.-Visualize-Metrics)
- [Real World Practices for Training and Regularization and PyTorch training template](PyTorch_training_template/index.ipynb)

---

## [CNN Building Blocks](#)
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

## [Modern CNN Architectures](#)
- LeNet
- AlexNet
- VGG
- U-Net, nnU-Net, 
- DeepLab, 
- MONAI
- [ResNet](modern_CNN_architectures/resnet.ipynb)
- [EfficientNet, MobileNet (for mobile)](modern_CNN_architectures/efficientnet.ipynb)
- [YOLO (You Only Look Once)](modern_CNN_architectures/yolo.ipynb)

## [Image Preprocessing & Augmentation Workflows](#)
- [Custom Dataset, ImageFolder, random_split, Subset](dataset/index.ipynb)  
- [DataLoader](dataloader/index.ipynb)  
- [Transforms, Pre-Processing, Data Augmentation](transform_pre_processing_augmentation/index.ipynb)  


---


## [**Sequence Modeling**](#)
[Encoder/ Decoder Architecture](encoder/index.ipynb)  
[Variational Autoencoders](machine_learning/nn/encoders.ipynb)  
- Teacher Forcing

---

## [**Attention & Transformers**](#) 
- [Transformer Architecture](transformer/attention.ipynb)
- Attention Mechanism (Additive, Dot-Product)
- Self-Attention & Multi-Head Attention
- [Positional Encoding](vit/index.ipynb#Step-6:-Add-Positional-Encoding)
- Transformer Encoder/Decoder
- [Vision Transformer (ViT)](vit/index.ipynb)
- [Transformer-Based Model for Monocular Visual](vit_monocular_vision/index.ipynb)
- Transformer Applications (BERT, GPT overview)

---




## [**3D Deep Learning**](#) 
- [PyTorch3D](PyTorch3D/index.ipynb)

---

## [**Advanced Topics & Research Trends**](#) 

- Autoencoders (Vanilla, Variational)
- GANs (Generator, Discriminator, Losses)
- [Diffusion Models (Denoising Score Matching)](diffusion_models/index.ipynb)
- Contrastive Learning (SimCLR, MoCo)
- Self-Supervised Learning
- Zero-shot & Few-shot Learning
- [Transfer learning, Fine tuning, Backbone, Neck, Head ](transfer_learning/transfer_learning.ipynb)  
- Large Language Models (LLMs)
- [Ensembling Models](model_ensembles/index.ipynb)
- [Flow Matching](flow_matching/index.ipynb)

---



## [**Multimodal Models**](multimodal_models)
- [Vision-Language Models (VLM)](multimodal_models/vision_language_models.ipynb)
- [Text-to-Image Models](multimodal_models/text_to_image.ipynb)
- [Audio-Visual Models](multimodal_models/audio_visual.ipynb)
- [Multimodal Transformers](multimodal_models/multimodal_transformers.ipynb)
- [Cross-Modal Retrieval](multimodal_models/cross_modal_retrieval.ipynb)

---

## [**LLM**](#)
- [A reading list that from Ilya Sutskever](https://arc.net/folder/D0472A20-9C20-4D3F-B145-D2865C0A9FEE)  
- [ollama](https://github.com/ollama/ollama)  
- [open-webui](https://github.com/open-webui/open-webui)  
- [LLM Visualization](https://bbycroft.net/llm)


## [**Production Engineering & MLOps**](#)
[Training stack & monitoring](training_stack_and_monitoring/index.ipynb)
- [Experiment Tracking (TensorBoard, WandB)](training_stack_and_monitoring/weights_and_biases_WandB/WandB.ipynb)
- [MLFlow](training_stack_and_monitoring/MLFlow/index.ipynb)
- [Training-time Monitoring](training_stack_and_monitoring/training_time_monitoring/index.ipynb)
- [TensorBoard & Model Debugging](training_stack_and_monitoring/tensorboard/index.ipynb)  
- [Visualizing Model Graphs & Gradients](training_stack_and_monitoring/torchviz_visualize_graphs/index.ipynb)


[Pre-deployment Quality & Validation](pre_deployment_quality_and_validation/index.ipynb)
- [Pre-release Quality Gates](pre_deployment_quality_and_validation/pre_release_quality_gates/index.ipynb)
- [Packaging for Inference](pre_deployment_quality_and_validation/packaging_for_inference/index.ipynb)
- [Model Deployment (ONNX, TorchScript)](pre_deployment_quality_and_validation/model_deployment_ONNX_torchscript/index.ipynb)
- Quantization & Pruning

[Deployment & Operations](deployment_and_operations/index.ipynb)
- [Release & Rollout Strategies](deployment_and_operations/release_rollout_strategies/index.ipynb)  
- [Production Monitoring](deployment_and_operations/production_monitoring/index.ipynb)  
- [A/B Testing](deployment_and_operations/ab_testing/index.ipynb)  
- Inference Optimization  

[Infrastructure & Best Practices](infrastructure_and_best_practices/index.ipynb)
- [Maximize GPU utilization](infrastructure_and_best_practices/maximize_GPU_utilization/index.ipynb)
- [Data Versioning](infrastructure_and_best_practices/data_versioning/index.ipynb)
- [Logging and Debugging](infrastructure_and_best_practices/logging_and_debugging/index.ipynb)
- [Project Structure & Best Practices](infrastructure_and_best_practices/project_structure/index.ipynb)