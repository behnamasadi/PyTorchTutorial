# PyTorchTutorial

This repository contains my snippets and sample codes for developing deep learning application with Pytorch.



![alt text](https://img.shields.io/badge/license-BSD-blue.svg)
![CI](https://github.com/behnamasadi/PyTorchTutorial/actions/workflows/ci.yml/badge.svg)
![GHCR](https://github.com/behnamasadi/PyTorchTutorial/actions/workflows/ghcr.yml/badge.svg)
[![GHCR Package](https://img.shields.io/badge/GHCR-Package-blue?logo=github&logoColor=white)](https://github.com/behnamasadi/PyTorchTutorial/pkgs/container/kaggle-projects)
![GitHub Issues or Pull Requests](https://img.shields.io/github/issues/behnamasadi/PyTorchTutorial)
<!-- ![GitHub Release](https://img.shields.io/github/v/release/behnamasadi/PyTorchTutorial) -->
![GitHub Repo stars](https://img.shields.io/github/stars/behnamasadi/PyTorchTutorial)
![GitHub forks](https://img.shields.io/github/forks/behnamasadi/PyTorchTutorial)


## Installation


```bash
conda create -n PyTorchTutorial python=3.12 -y
. "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate PyTorchTutorial
```

### Install PyTorch

For a machine with a working NVIDIA driver and `nvidia-smi`, install the current CUDA build from the official PyTorch wheel index:

```bash
python -m pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```


```bash
pip install jupyterlab matplotlib seaborn scikit-learn pydot torchviz mlflow timm opencv-python albumentations tqdm tensorboard wandb kagglehub pytorch-lightning shap
```

For the RAG notebook (embeddings, reranking, a persistent vector store):

```bash
pip install chromadb sentence-transformers
```

Generation runs on whatever you serve locally; the notebook uses [Ollama](https://ollama.com):

```bash
ollama pull qwen3:30b-a3b-instruct-2507-q4_K_M   # generator
ollama pull qwen3-embedding:0.6b                 # embedder
```

`monai[all]` is intentionally not installed by default because it is large and pulls many optional dependencies. Install it only if you need the medical imaging notebooks:

```bash
pip install "monai[all]"
```

### Optional system packages

If you want to view generated graphviz `.dot` files:

```bash
sudo apt-get install graphviz
sudo apt-get install xdot
```

### Repo location

If you want the env-local `src` symlink that points to this repo:

```bash
ln -s /home/$USER/workspace/PyTorchTutorial /home/$USER/anaconda3/envs/PyTorchTutorial/src
```

### Updating packages

Do not use `conda update --all` for this env. It tends to churn the full stack and is a common way to break a working PyTorch/CUDA setup.

Upgrade only what you actually need, for example:

```bash
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```


## [**PyTorch Fundamentals**](#)

- [PyTorch Tensor Basics & Data Types](data_types/index.ipynb)
- [Einsum Operator](einsum_operator/index.ipynb)
- [Grad Package](grad_package/)  
  - [Computational Graph](grad_package/grad.ipynb#Computational-Graph)  
  - [Autograd](grad_package/grad.ipynb#Autograd)  
  - [Dynamic Computational Graph](grad_package/grad.ipynb#)  
  - [Detach](grad_package/grad.ipynb#detach)  
  - [Exclusion from the DAG](grad_package/grad.ipynb#Exclusion-from-the-DAG)  
  - [Leaf Tensor](grad_package/grad.ipynb#Leaf)  
  - [No Grad](grad_package/grad.ipynb#no_grad())  
  - [Zero Grad](grad_package/grad.ipynb#zero_grad)
  - [requires_grad](grad_package/grad.ipynb#requires_grad)
- [Model Saving & Loading (Serialization)](serialization_saving_loading/index.ipynb)  
- [Parameters Registration, ModuleList](modulelist/index.ipynb)

## [Neural Network Basics](#)

- [Neural Networks, Manifolds, and Topology](https://colah.github.io/posts/2014-03-NN-Manifolds-Topology/)  
- [Back Propagation](backpropagation/index.ipynb)  
- [Activation Functions](activation_functions/index.ipynb)  
- [Loss Functions](loss_functions/index.ipynb)  
- [Inductive Bias](inductive_bias/index.ipynb)  

## [Training Process](#)

- [Optimizer Package](optim_package/optimizers.ipynb)  
- [Learning Rate & Learning Rate Scheduler Schedulers](optim_package/learning_rate_scheduler.ipynb)  
- [Regularization](regularization/index.ipynb)
- [Dropout Layers](dropout_layers/index.ipynb)
- [Normalization](batch_layer_instance_group_normalization/index.ipynb)
  - [Batch Normalization](batch_layer_instance_group_normalization/batch_normalization.ipynb)
  - [Layer Normalization](batch_layer_instance_group_normalization/layer_normalization.ipynb)
  - [Instance Normalization](batch_layer_instance_group_normalization/instance_normalization.ipynb)
  - [Group Normalization](batch_layer_instance_group_normalization/group_normalization.ipynb)
- [Weight Initialization Strategies](weight_initialization/index.ipynb)
- [Evaluation vs Training Mode (Learning Monitoring)](learning_monitoring/index.ipynb)
  - [Training, Validation, and Test Set](learning_monitoring/index.ipynb#Training-and-Validation-set)
  - [Monitor for Overfitting](learning_monitoring/index.ipynb#1.-Monitor-for-Overfitting)
  - [Early Stopping](learning_monitoring/index.ipynb#2.-Implement-Early-Stopping)
  - [Visualize Metrics](learning_monitoring/index.ipynb#4.-Visualize-Metrics)
- [Real World Practices for Training and Regularization and PyTorch training template](PyTorch_training_template/index.ipynb)
- [Function Approximation](function_approximation/function_approximation.py)

---

## [CNN Building Blocks](#)

- [Convolution, Cross-Correlation, Transposed Convolution (Deconvolution), 1x1 Convolution](conv/cross_correlation_convolution.ipynb#1.-Cross-Correlation)
- [Shape of Output](conv/cross_correlation_convolution.ipynb#4.Shape-of-the-Convolution-Output)
- [RGB Image Convolution](conv/cross_correlation_convolution.ipynb#5.Convolution-in-RGB-Images)
- [Convolution as Matrix Multiplication](conv/cross_correlation_convolution.ipynb#Convolution-as-Matrix-Multiplication)
- [Conv2d class vs conv2d function](conv/cross_correlation_convolution.ipynb#PyTorch-Conv2d-class-vs-conv2d-function)
- [Unfold/ fold](unfold/index.ipynb)
- [Padding, Stride, Dilation](conv/cross_correlation_convolution.ipynb#4.Shape-of-the-Convolution-Output)
- [Pooling (Max, Average, Adaptive), Order of Relu and  Max Pooling](pooling/index.ipynb)
- [Feature Map](conv/cross_correlation_convolution.ipynb#8.-Feature-Map)
- [Convolution is translation-Equivariant, not Translation-Invariant](conv/cross_correlation_convolution.ipynb#11.-Convolution-is-translation-Equivariant,-not-Translation-Invariant)
- [Visualize Architecture of Neural Network](https://github.com/ashishpatel26/Tools-to-Design-or-Visualize-Architecture-of-Neural-Network)

---

## [Modern Vision Architectures](#)

#### [**CNN Architectures**](#)

- [VGG](cnn_architectures/vgg.ipynb)
- [ResNet](cnn_architectures/resnet.ipynb)
- [RegNet](cnn_architectures/regnet.ipynb)
- [EfficientNet](cnn_architectures/efficientnet.ipynb)
- [MobileNet](cnn_architectures/mobilenet.ipynb)
- [ConvNeXt](cnn_architectures/convnext.ipynb)
- [DenseNet](cnn_architectures/densenet.ipynb)
- [Inception](cnn_architectures/inception.ipynb)
---

#### [**Image Segmentation**](#)

- [Semantic, Instance, and Panoptic Segmentation](segmentation/index.ipynb)
    -[Panoptic Architectures](segmentation/index.ipynb)
- [U-Net](segmentation/unet.ipynb)
- [DeepLab](segmentation/deeplab.ipynb)
- [SAM 3](segmentation/SAM3.ipynb)
- [Saliency Detection](segmentation/saliency_detection.ipynb)

---

#### [**Object Detection**](#)

- [Object Detection Evaluation Metrics](object_detection/object_detection_evaluation_metrics.ipynb)
- [YOLO (You Only Look Once)](object_detection/yolo.ipynb)
- [Faster R-CNN](object_detection/faster_rcnn.ipynb)
- [SSD (Single Shot Detector)](object_detection/ssd.ipynb)
- [RetinaNet](object_detection/retinanet.ipynb)
- [DETR](object_detection/detr.ipynb)
- [Mask R-CNN - Instance segmentation + detection](object_detection/mask_rcnn.ipynb)

---

## [**Medical Imaging**](#)

- [MONAI (Medical Open Network for AI)](medical_imaging/monai.ipynb)
- [nnU-Net (Biomedical Image Segmentation)](medical_imaging/nnunet.ipynb)

---



## [Image Preprocessing & Augmentation Workflows](#)

- [DataLoader, Custom Dataset, TensorDataset, ImageFolder, random_split, Subset](dataset/index.ipynb)  
- [Transforms, Pre-Processing](transform_pre_processing_augmentation/transform_pre_processing.ipynb)  
- [Data Augmentation](transform_pre_processing_augmentation/augmentation.ipynb)  
- [OpenCV and PIL Image Format](opencv_pil/index.ipynb)

---

## [**Attention & Transformers**](#)

- [Transformer Architecture](transformer/attention.ipynb)
- [Relative Positional Encoding](transformer/relative_positional_encoding.ipynb)
- [Vision Transformer](transformer/vit.ipynb)
- [Swin Transformer](transformer/swin.ipynb)
- [DINO](transformer/DINO.ipynb)  
- [CLIP](transformer/CLIP.ipynb)  
- [DeiT](transformer/DeiT.ipynb)  
- [Pyramid Vision Transformer](transformer/pvt_pyramid_vision_transformer.ipynb)
- [Feature Pyramid Network (FPN)](feature_pyramid_network/index.ipynb)
- [Temporal Transformer](transformer/temporal_transformer.ipynb)

---

## [**Foundation Models**](#)

General-purpose, self-supervised backbones that are frozen and reused across many downstream tasks (classification, retrieval, segmentation, depth, correspondence) via lightweight heads.

- [DINOv3 — one backbone, many outputs](foundation_models/DINOv3.ipynb)
  - [The three outputs: CLS, patch, register tokens](foundation_models/DINOv3.ipynb)
  - [Global embedding → retrieval / classification / clustering](foundation_models/DINOv3.ipynb)
  - [Patch features → segmentation, depth, correspondence](foundation_models/DINOv3.ipynb)
  - [Depth estimation script (CLI)](foundation_models/scripts/dinov3.py)

---

## [**Advanced Topics & Research Trends**](#)

- [Encoder/ Decoder Architecture](encoder/index.ipynb)  
- [Variational Autoencoders](encoder/index.ipynb#Variational-Autoencoders)  
- [Diffusion Models (Denoising Score Matching)](diffusion_models/index.ipynb)
- [Contrastive Learning](contrastive_learning/index.ipynb)
- [Zero-shot & Few-shot Learning](zero_shot_few_shot_learning/index.ipynb)
- [Transfer learning, Fine tuning, Backbone, Neck, Head](transfer_learning_fine_tuning/transfer_learning_fine_tuning.ipynb)  
- [Fine-tuning Vision Transformers: frozen head, LLRD, LoRA](transfer_learning_fine_tuning/vit_fine_tuning.ipynb)
  - [Why ViTs are fine-tuned differently from CNNs](transfer_learning_fine_tuning/vit_fine_tuning.ipynb)
  - [Frozen backbone + DPT-style head (and the detail stem for thin structures)](transfer_learning_fine_tuning/vit_fine_tuning.ipynb)
  - [Full fine-tune with layer-wise learning-rate decay (LLRD)](transfer_learning_fine_tuning/vit_fine_tuning.ipynb)
  - [LoRA / parameter-efficient tuning with `peft`](transfer_learning_fine_tuning/vit_fine_tuning.ipynb)
  - [Checklist before a run: patch size, normalisation, tokens, partial labels](transfer_learning_fine_tuning/vit_fine_tuning.ipynb)
  - [All three strategies in one runnable script](transfer_learning_fine_tuning/scripts/vit_ft_examples.py)
- [Ensembling Models](ensembling_models/index.ipynb)
- [Flow Matching](flow_matching/index.ipynb)
- [Making Network Deterministic](deterministic_network/index.ipynb)
- [Knowledge Distillation](knowledge_distillation/index.ipynb)
- [Neural Architecture Search (NAS), and Design Spaces](neural_architecture_search_design_spaces/index.ipynb)
- [PyTorch Image Models ( timm )](timm_image_model/index.ipynb)
- [Stochastic Depth](stochastic_depth/index.ipynb)
- [3D CNN](3D_CNN/index.ipynb)
- [Squeeze-and-Excitation Networks (SENet)](squeeze_and_excitation_SE/squeeze_and_excitation_networks_SENet.ipynb)
- [Convolutional Block Attention Module (CBAM)](CBAM/convolutional_block_attention_module_CBAM.ipynb)
- [MBConv](MBConv/index.ipynb)
- [Fused-MBConv](Fused-MBConv/index.ipynb)
- [Double Descent](bias_variance_tradeoff_double_descent/index.ipynb)
- [Receptive Field](receptive_field/index.ipynb)
- [Universal Approximation Theorem](universal_approximation_theorem/index.ipynb)
- [Degrading Problem in Deep Learning](degrading/index.ipynb)
- [PyTorch Lightning](pytorch_lightning/index.ipynb)
- [Gradient Clipping](gradient_clipping/index.ipynb)

---

## [**Multimodal Models**](multimodal_models/index.ipynb)

- [Vision-Language Models (VLM)](multimodal_models/vision_language_models.ipynb)
- [Text-to-Image Models](multimodal_models/text_to_image.ipynb)
- [Audio-Visual Models](multimodal_models/audio_visual.ipynb)
- [Multimodal Transformers](multimodal_models/multimodal_transformers.ipynb)
- [Cross-Modal Retrieval](multimodal_models/cross_modal_retrieval.ipynb)

---

## [**Retrieval-Augmented Generation (RAG)**](rag/index.ipynb)

Adapting a pretrained model *without* training it: knowledge is retrieved at question time and put into the prompt, while the model stays frozen.

- [RAG — Retrieval-Augmented Generation](rag/index.ipynb)
  - [Fine-tuning vs RAG: what changes, what it costs](rag/index.ipynb)
  - [Chunking, embeddings, and the vector index](rag/index.ipynb)
  - [Hybrid retrieval (dense + BM25 + RRF) and cross-encoder reranking](rag/index.ipynb)
  - [Prompt assembly, citations, and evaluating retrieval vs generation](rag/index.ipynb)
  - [Visual RAG: k-NN over frozen DINOv3 embeddings, VLM exemplar prompting](rag/index.ipynb)
  - [Persisting the index with Chroma: cosine space, content-hash ids, metadata filters](rag/index.ipynb)
  - [Local generation with Ollama: `num_ctx`, VRAM budget, Qwen3 embeddings](rag/index.ipynb)
  - [A whole RAG pipeline in one file (no framework)](rag/scripts/mini_rag.py)

---

## [**LLM**](#)

- [A reading list that from Ilya Sutskever](https://arc.net/folder/D0472A20-9C20-4D3F-B145-D2865C0A9FEE)  
- [ollama](https://github.com/ollama/ollama)  
- [open-webui](https://github.com/open-webui/open-webui)  
- [LLM Visualization](https://bbycroft.net/llm)
- [LLM Transparency Tool](https://github.com/facebookresearch/llm-transparency-tool/?tab=readme-ov-file)

---

## [XAI (Explainable Artificial Intelligence)](xai/index.ipynb)

- [Model-agnostic (post-hoc explanations)](#)
  - [SHAP (SHapley Additive exPlanations)](xai/shap.ipynb)
  - [Saliency Maps / Grad-CAM](xai/grad-cam.ipynb)

---

## [**Production Engineering & MLOps**](#)

[Experiment Tracking & Monitoring](training_stack_and_monitoring/index.ipynb)

- [Weights & Biases](weights_and_biases/index.ipynb)
- [MLFlow](training_stack_and_monitoring/MLFlow/index.ipynb)
- [TensorBoard](training_stack_and_monitoring/tensorboard/index.ipynb)  
- [Neptune](neptune/index.ipynb)
- [Training-time Monitoring](training_stack_and_monitoring/training_time_monitoring/index.ipynb)
- [Visualizing Model Graphs & Gradients](training_stack_and_monitoring/torchviz_visualize_graphs/index.ipynb)

[Pre-deployment Quality & Validation](pre_deployment_quality_and_validation/index.ipynb)

- [Pre-release Quality Gates](pre_deployment_quality_and_validation/pre_release_quality_gates/index.ipynb)
- [Packaging for Inference](pre_deployment_quality_and_validation/packaging_for_inference/index.ipynb)
- [Model Deployment (ONNX, TorchScript)](pre_deployment_quality_and_validation/model_deployment_ONNX_torchscript/index.ipynb)
- [TensorRT](pre_deployment_quality_and_validation/tensor_rt.ipynb)
- Quantization & Pruning

[Deployment & Operations](deployment_and_operations/index.ipynb)

- [Release & Rollout Strategies](deployment_and_operations/release_rollout_strategies/index.ipynb)  
- [Production Monitoring](deployment_and_operations/production_monitoring/index.ipynb)  
- [A/B Testing](deployment_and_operations/ab_testing/index.ipynb)  
- Inference Optimization  

## [Local LLM Inference](#)

- [llama.cpp, GGUF & Quantization — Running Real LLMs on One GPU](llama_cpp_gguf/index.ipynb)
  - [The GGUF file format, parsed from scratch in pure Python](llama_cpp_gguf/index.ipynb)
  - [PyTorch (`.safetensors` / `.bin` / `.pt`) and TensorFlow → GGUF](llama_cpp_gguf/index.ipynb)
  - [Loading huge weights: sharding, `mmap`, lazy conversion](llama_cpp_gguf/index.ipynb)
  - [Block quantization: Q4_0, K-quants, IQ-quants, imatrix — implemented in NumPy](llama_cpp_gguf/index.ipynb)
  - [Decoding real 4-bit Q4_K weights off disk](llama_cpp_gguf/index.ipynb)
  - [VRAM math & what actually fits on an RTX 3090 (24 GB)](llama_cpp_gguf/index.ipynb)
  - [Reading model names: `qwen2.5-coder:32b`, `30B-A3B`, `Q4_K_M`](llama_cpp_gguf/index.ipynb)
  - [The open-weight model landscape: Qwen, DeepSeek, GLM, Gemma, Mistral, gpt-oss, Kimi](llama_cpp_gguf/index.ipynb)
    - [The full Qwen family tree — text, coder, VL, embedding, reranker, omni, guard, TTS, image](llama_cpp_gguf/index.ipynb)
    - [What each lab is actually best at, and what to download for a 3090](llama_cpp_gguf/index.ipynb)
    - [What does *not* fit (Kimi K3, DeepSeek-V4) and when MoE CPU offload rescues it](llama_cpp_gguf/index.ipynb)
  - [Hybrid attention (Gated DeltaNet) and why it shrinks the KV cache 4×](llama_cpp_gguf/index.ipynb)
  - [llama.cpp without a daemon, plus Ollama and aider in practice](llama_cpp_gguf/index.ipynb)

## [GPU Optimization & Performance](#)

- [Maximize GPU Utilization](gpu_optimization_and_performance/maximize_gpu_utilization/index.ipynb)
- [Gradient Accumulation](gpu_optimization_and_performance/gradient_accumulation/)
- [AMP Automatic Mixed Precision (amp)](gpu_optimization_and_performance/amp_mixed_precision/index.ipynb)
- [Gradient Checkpointing](gpu_optimization_and_performance/gradient_check_pointing/)
- [Memory Monitoring & Management](gpu_optimization_and_performance/memory_monitoring_management)
- [Optimal Batch Size Selection](gpu_optimization_and_performance/optimal_batch_size_selection)
- [Efficient Data Loading](gpu_optimization_and_performance/efficient_data_loading/index.ipynb)
- [GPU Memory Optimization Techniques](gpu_optimization_and_performance)
- [Clear Unused Variables & Cache](gpu_optimization_and_performance/clear_unused_variables_cache/index.ipynb)


---

## [Infrastructure & Best Practices](infrastructure_and_best_practices/index.ipynb)

- [Data Versioning](infrastructure_and_best_practices/data_versioning/index.ipynb)
- [Logging and Debugging](infrastructure_and_best_practices/logging_and_debugging/index.ipynb)
- [Project Structure & Best Practices](infrastructure_and_best_practices/project_structure/index.ipynb)
- [Running Your PyTorch Projects on RunPod Using a Single Docker Image and GHCR](infrastructure_and_best_practices/runpod-ghcr.ipynb)
- [GitHub Actions CI Setup, CUDA with CPU Fallback](infrastructure_and_best_practices/github_action_ci.ipynb)
---

## [Kaggle](#)

- [Kaggle Dataset Downloader & Management](kaggle_structure/index.ipynb)
  - [Automated Kaggle dataset download with symlinks](kaggle_structure/index.ipynb)
  - [Finding & searching datasets (CLI & web)](kaggle_structure/index.ipynb)
  - [Running in Kaggle notebooks vs local/RunPod](kaggle_structure/index.ipynb)
  - [Kaggle GPU specifications & optimization](kaggle_structure/index.ipynb)
  - [Medical imaging, computer vision, 3D vision dataset examples](kaggle_structure/index.ipynb)

## [Datasets](#)
- [Waymo Open Dataset](https://waymo.com/open/)
- [City Scapes Dataset](https://www.cityscapes-dataset.com/)
- [KITTI](https://www.cvlibs.net/datasets/kitti/) — CC BY-NC-SA (share-alike, non-commercial)
- [A2D2 (Audi)](https://www.a2d2.audi/a2d2/en.html) — CC BY-ND
- [ADE20K](https://ade20k.csail.mit.edu/) — BSD-3, great for scene segmentation
- [COCO / COCO-Stuff](https://cocodataset.org/) — CC BY
- [Comma2k19](https://github.com/commaai/comma2k19) — MIT

---

## [Free Books and Online Courses](#)

- [Learning Deep Representations of Data Distributions](https://ma-lab-berkeley.github.io/deep-representation-learning-book/)
- [The Principles of Diffusion Models](https://www.arxiv.org/abs/2510.21890)
- [3D Scanning & Motion Capture (TUM-Matthias Niessner)](https://niessner.github.io/3DScanning/)
- [Machine Learning for 3D Data](https://3dml.kaist.ac.kr/)
- [Neural Radiance Fields | NeRF ](https://www.youtube.com/watch?v=Q1zqf5tfeJw)
- [The Principles of Diffusion Models](https://arxiv.org/pdf/2510.21890)  
---
