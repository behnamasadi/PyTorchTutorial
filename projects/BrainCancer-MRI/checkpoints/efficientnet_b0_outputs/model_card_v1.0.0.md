# Brain Cancer MRI Classification Model Card

## Model Information
- **Model Name**: efficientnet_b0
- **Version**: 1.0.0
- **Description**: First production model with 92.30% test accuracy
- **Registration Date**: 2025-08-10T22:44:58.038947
- **Model Type**: Medical AI - Brain Tumor Classification

## Performance Metrics

### Overall Performance
- **Test Accuracy**: 0.9230 (92.30%)
- **Macro F1-Score**: 0.9238
- **Macro Precision**: 0.9267
- **Macro Recall**: 0.9230

### Per-Class Performance

#### Glioma
- **Precision**: 0.9745
- **Recall**: 0.9144
- **F1-Score**: 0.9435
- **Support**: 292 samples

#### Meningioma
- **Precision**: 0.8550
- **Recall**: 0.9371
- **F1-Score**: 0.8942
- **Support**: 302 samples

#### Pituitary
- **Precision**: 0.9507
- **Recall**: 0.9175
- **F1-Score**: 0.9338
- **Support**: 315 samples

## Medical AI Validation
- **Medical Threshold Passed**: ✅ YES
- **Deployment Ready**: ✅ YES
- **Sensitivity Warnings**: 0
- **Recommendations**: 0

## Inference Performance
- **Average Inference Time**: 0.64ms per sample
- **Batch Processing Time**: 0.020s per batch
- **Total Test Samples**: 909

## Model Files
- **efficientnet_b0_model.pt**: 16.02 MB
- **efficientnet_b0_model.onnx**: 15.29 MB

## Training Configuration
- **Epoch**: 6
- **Validation Accuracy**: 0.9162995594713657
- **Checkpoint Size**: 15.62 MB

## Usage Instructions
1. Load the model using the appropriate framework (PyTorch for .pt, ONNX Runtime for .onnx)
2. Preprocess input images to match the expected input size
3. Run inference and post-process the outputs
4. Apply medical validation checks before clinical use

## Medical AI Compliance
This model has been validated for medical AI deployment with:
- Accuracy threshold validation (≥85%)
- Sensitivity analysis for tumor detection
- Clinical deployment readiness assessment
- Complete audit trail for regulatory compliance

---
*Generated automatically by Brain Cancer MRI Model Registry*
