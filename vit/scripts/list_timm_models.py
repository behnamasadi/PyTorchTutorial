import timm

print("=" * 80)
print("HOW TO LIST TIMM MODELS")
print("=" * 80)

# 1. List ALL available models (warning: there are 1000+)
print("\n1. Total number of models in timm:")
all_models = timm.list_models()
# print(f"   Total: {len(all_models)} models")

# 2. List only pretrained models
print("\n2. Pretrained models:")
pretrained_models = timm.list_models(pretrained=True)
print(f"   Total pretrained: {len(pretrained_models)} models")

# 3. Search for specific model types
print("\n3. Search for specific models:")
print("\n   a) Vision Transformers (ViT):")
vit_models = timm.list_models('*vit*', pretrained=True)
print(f"      Found {len(vit_models)} ViT models")
for i, model in enumerate(vit_models[:5]):
    print(f"      - {model}")
if len(vit_models) > 5:
    print(f"      ... and {len(vit_models) - 5} more")

print("\n   b) DeiT models:")
deit_models = timm.list_models('deit*', pretrained=True)
print(f"      Found {len(deit_models)} DeiT models")
for model in deit_models:
    print(f"      - {model}")

print("\n   c) Swin Transformer:")
swin_models = timm.list_models('swin*', pretrained=True)
print(f"      Found {len(swin_models)} Swin models")
for i, model in enumerate(swin_models[:5]):
    print(f"      - {model}")
if len(swin_models) > 5:
    print(f"      ... and {len(swin_models) - 5} more")

print("\n   d) ResNet models:")
resnet_models = timm.list_models('resnet*', pretrained=True)
print(f"      Found {len(resnet_models)} ResNet models")
for i, model in enumerate(resnet_models[:5]):
    print(f"      - {model}")
if len(resnet_models) > 5:
    print(f"      ... and {len(resnet_models) - 5} more")

print("\n   e) EfficientNet models:")
effnet_models = timm.list_models('efficientnet*', pretrained=True)
print(f"      Found {len(effnet_models)} EfficientNet models")
for i, model in enumerate(effnet_models[:5]):
    print(f"      - {model}")
if len(effnet_models) > 5:
    print(f"      ... and {len(effnet_models) - 5} more")

print("\n   f) RegNet models:")
regnet_models = timm.list_models('regnet*', pretrained=True)
print(f"      Found {len(regnet_models)} RegNet models")
for i, model in enumerate(regnet_models[:8]):
    print(f"      - {model}")
if len(regnet_models) > 8:
    print(f"      ... and {len(regnet_models) - 8} more")

# 4. Search with multiple patterns
print("\n4. Search with filters:")
print("\n   Transformer models (various types):")
transformer_keywords = ['vit', 'deit', 'swin', 'beit', 'mae']
for keyword in transformer_keywords:
    models = timm.list_models(f'*{keyword}*', pretrained=True)
    if models:
        print(f"   - {keyword.upper()}: {len(models)} models")

# 5. Get model information
print("\n5. Get detailed model info:")
print("\n   Example: Getting info for 'resnet50'")
try:
    model = timm.create_model('resnet50', pretrained=False)
    print(f"   - Architecture: ResNet-50")
    print(f"   - Input size: {model.default_cfg.get('input_size', 'N/A')}")
    print(f"   - Number of classes: {model.num_classes}")
    num_params = sum(p.numel() for p in model.parameters())
    print(f"   - Parameters: {num_params:,}")
except Exception as e:
    print(f"   Error: {e}")

print("\n" + "=" * 80)
print("USEFUL TIMM FUNCTIONS:")
print("=" * 80)
print("""
1. timm.list_models()                      # List all models
2. timm.list_models(pretrained=True)       # List pretrained models only
3. timm.list_models('resnet*')             # Wildcard search
4. timm.list_models('*vit*')               # Search containing 'vit'
5. timm.create_model('model_name')         # Create a model
6. timm.list_models(filter='keyword')      # Alternative filter syntax

Examples:
  timm.list_models('vit_*_patch16_224')    # Specific pattern
  timm.list_models('*convnext*')           # ConvNeXt models
  timm.list_models('*mobile*')             # MobileNet variants
""")

print("=" * 80)
print("TIP: Use interactive Python to explore:")
print("=" * 80)
print("""
>>> import timm
>>> models = timm.list_models('swin*', pretrained=True)
>>> for m in models: print(m)
""")

print("\n✅ Done! You can now find any model you need.")

print("\n" + "=" * 80)
print("MODEL NAMING CONVENTIONS & PARAMETERS")
print("=" * 80)

print("\nExample: 'regnetv_040.ra3_in1k'")
print("\nNaming breakdown:")
print("  • 'regnetv'    = Model architecture (RegNet variant V)")
print("  • '040'        = Model size/capacity (4.0 GFLOPs)")
print("  • 'ra3'        = Training recipe/augmentation (RandAugment v3)")
print("  • 'in1k'       = Dataset (ImageNet-1k)")

print("\nCommon naming patterns:")
print("  • Model sizes: 'tiny', 'small', 'base', 'large', or numbers like '040', '160'")
print("  • Patch sizes: 'patch16', 'patch32' (for transformers)")
print("  • Resolution: '224', '384', '512' (input image size)")
print("  • Datasets: 'in1k' (ImageNet-1k), 'in21k' (ImageNet-21k)")
print("  • Training: 'ra' (RandAugment), 'dist' (distilled), 'ft' (fine-tuned)")

print("\n" + "-" * 80)
print("Getting model parameters:")
print("-" * 80)

# Example: regnetv_040.ra3_in1k
try:
    print("\nAnalyzing: regnetv_040.ra3_in1k")
    model = timm.create_model('regnetv_040.ra3_in1k', pretrained=False)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)

    print(f"  ✓ Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print(
        f"  ✓ Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    print(f"  ✓ Input size: {model.default_cfg.get('input_size', 'N/A')}")
    print(f"  ✓ Number of classes: {model.num_classes}")
    print(f"  ✓ Architecture: {model.__class__.__name__}")

except Exception as e:
    print(f"  ✗ Error: {e}")

# Compare different model sizes
print("\n" + "-" * 80)
print("Comparing different RegNet models:")
print("-" * 80)

regnet_examples = ['regnetv_040.ra3_in1k',
                   'regnetv_064.ra3_in1k', 'regnetv_080.ra3_in1k']
for model_name in regnet_examples:
    try:
        model = timm.create_model(model_name, pretrained=False)
        params = sum(p.numel() for p in model.parameters())
        print(f"  {model_name:30s} → {params/1e6:6.2f}M parameters")
    except:
        print(f"  {model_name:30s} → Not available")

print("\n" + "-" * 80)
print("How to get model info for any model:")
print("-" * 80)
print("""
import timm

# Create model (pretrained=False is faster for just checking params)
model = timm.create_model('model_name', pretrained=False)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {total_params:,} ({total_params/1e6:.2f}M)")

# Get configuration
print(f"Input size: {model.default_cfg['input_size']}")
print(f"Classes: {model.num_classes}")
""")
