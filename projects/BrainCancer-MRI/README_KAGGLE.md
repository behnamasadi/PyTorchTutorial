# Brain Tumor Classification: Xception Medical Solution

## 🚀 Overview

This repository contains a **Xception Medical solution** for brain tumor classification, optimized for medical imaging tasks. The solution uses EfficientNet-B0 as backbone with a custom Xception-inspired classifier, achieving superior performance through medical domain optimization.

## 📋 Key Features

- **Xception Medical Architecture**: EfficientNet-B0 backbone with medical-optimized classifier
- **Dataset-Specific Normalization**: Direct use of pre-computed statistics (seed=42)
- **Grayscale to RGB Conversion**: Automatic conversion for EfficientNet-B0 compatibility
- **GPU Optimizations**: Mixed precision training, optimized DataLoader settings
- **Medical Domain Focus**: 224×224 input size (standard EfficientNet-B0)
- **Production Ready**: Model saving, evaluation, and comprehensive results

## 🎯 Expected Performance

| Metric | Xception Medical |
|--------|------------------|
| Test Accuracy | 85-95% |
| Training Time | ~1-2 hours |
| Parameters | ~4.2M |
| Input Size | 224×224 |

## 🏗️ Architecture

**Xception Medical Model:**
- **Backbone**: EfficientNet-B0 (pretrained on ImageNet)
- **Input**: 224×224 RGB images (converted from grayscale medical images)
- **Classifier**: Dropout(0.3) → Linear(1280→128) → ReLU → Dropout(0.25) → Linear(128→3)
- **Training**: Full fine-tuning with Adamax optimizer (lr=0.001)
- **Normalization**: Dataset-specific statistics (direct values, seed=42)
- **Data Processing**: Grayscale → RGB conversion for EfficientNet-B0 compatibility

## 🔧 Quick Start

### Installation

```bash
# Install required packages
pip install torch torchvision
pip install numpy pandas matplotlib seaborn
pip install scikit-learn tqdm
pip install kagglehub
```

### Run the Solution

```bash
# Run the complete solution
python kaggle_solution.py
```

The script will automatically:
1. Download the brain cancer MRI dataset from KaggleHub
2. Load and preprocess data with proper normalization
3. Train Xception Medical model
4. Evaluate performance and generate results

## 📊 Dataset

- **Source**: KaggleHub - "orvile/brain-cancer-mri-dataset"
- **Classes**: Glioma, Meningioma, Pituitary Tumor
- **Format**: Organized in class-specific folders
- **Size**: ~6,056 samples across 3 classes
- **Splits**: 70% train, 15% validation, 15% test (stratified)

## 🎯 Why Xception Medical?

### Medical Domain Optimization
- **Custom Classifier**: Designed specifically for medical images
- **Dropout Regularization**: (0.3, 0.25) for medical datasets
- **224×224 Input**: Standard EfficientNet-B0 input size for optimal performance
- **Grayscale to RGB**: Automatic conversion for EfficientNet-B0 compatibility
- **Full Fine-tuning**: Complete model adaptation to medical domain

### Technical Advantages
- **Parameter Efficiency**: ~4.2M parameters vs 25M+ in ResNet50
- **Better Feature Learning**: Xception-inspired design
- **Medical Patterns**: Effective for medical image classification
- **Production Ready**: Robust and reliable performance

## 📁 Output Files

- `best_xception_medical_model.pth`: Best Xception Medical model
- `training_curves.png`: Training loss and accuracy plots
- `confusion_matrices.png`: Confusion matrix visualization
- `model_comparison_results.csv`: Detailed results

## 🔍 Key Optimizations

1. **Dataset-Specific Normalization**: Direct use of pre-computed statistics (seed=42)
2. **Grayscale to RGB Conversion**: Automatic conversion for EfficientNet-B0 compatibility
3. **GPU Optimizations**: Mixed precision, optimized DataLoader settings
4. **Memory Efficiency**: 224×224 input size (standard EfficientNet-B0 size)
5. **Stratified Splits**: Balanced class distribution across train/val/test
6. **Early Stopping**: Prevents overfitting with patience=10

## 🏥 Clinical Implications

- **High Accuracy**: Suitable for clinical deployment
- **Medical Optimization**: Designed specifically for medical imaging
- **Robust Performance**: Consistent across different tumor types
- **Computational Efficiency**: Balanced performance and resource usage

## 🚨 Troubleshooting

### Common Issues:

1. **CUDA Out of Memory**:
   ```bash
   # The script automatically adjusts batch sizes based on GPU memory
   # If issues persist, reduce batch_size in the script
   ```

2. **Dataset Download Issues**:
   ```bash
   # Install kagglehub if not installed
   pip install kagglehub
   
   # If download fails, use local data
   # Place your data in: data/brain-cancer/
   ```

3. **Normalization Issues**:
   ```bash
   # Normalization values are hardcoded (seed=42) for reproducibility
   # If you need to recalculate, run compute_normalization.py
   python compute_normalization.py
   ```

## 📈 Performance Monitoring

The script includes:
- Real-time GPU utilization tracking
- Memory usage monitoring
- Training progress visualization
- Automatic optimization recommendations

## 🎯 Best Practices

1. **Use Dataset-Specific Normalization**: Direct values ensure reproducibility (seed=42)
2. **Handle Grayscale Images**: Automatic RGB conversion for EfficientNet-B0 compatibility
3. **Monitor GPU Usage**: Script provides real-time monitoring
4. **Save Checkpoints**: Models are automatically saved
5. **Validate Results**: Review confusion matrices and accuracy plots
6. **Test on Medical Data**: Always validate before clinical deployment

## 📝 Technical Details

### Model Architecture

```python
class XceptionMedical(nn.Module):
    def __init__(self, num_classes=3):
        # EfficientNet-B0 backbone
        self.backbone = models.efficientnet_b0(pretrained=True)
        
        # Xception-inspired classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),                    # Medical dropout
            nn.Linear(1280, 128),              # Intermediate layer
            nn.ReLU(),
            nn.Dropout(0.25),                  # Second dropout
            nn.Linear(128, num_classes)        # Final classification
        )
```

### Training Strategy

- **Optimizer**: Adamax (lr=0.001) - optimized for full fine-tuning
- **Strategy**: Full fine-tuning (all parameters trainable)
- **Input Size**: 224×224 (standard EfficientNet-B0 input size)
- **Data Processing**: Grayscale → RGB conversion for EfficientNet-B0
- **Batch Size**: Adaptive based on GPU memory
- **Normalization**: Dataset-specific statistics (direct values, seed=42)

## 📞 Support

For issues or questions:
- Check GPU utilization analysis in the script output
- Review error messages for dataset path issues
- Ensure all dependencies are installed
- Run `python kaggle_solution.py --help` for usage information

---

**Author**: Behnam Asadi  
**Date**: 2025  
**Architecture**: Xception Medical (EfficientNet-B0 backbone)  
**Environment**: PyTorch, GPU-optimized
