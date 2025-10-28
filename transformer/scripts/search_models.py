#!/usr/bin/env python
"""
Quick script to search for timm models
Usage: python search_models.py <search_term>
Example: python search_models.py vit
         python search_models.py resnet50
"""

import sys
import timm

def search_models(query='', show_pretrained_only=True):
    """Search for models in timm library"""
    
    if query:
        pattern = f'*{query}*'
        models = timm.list_models(pattern, pretrained=show_pretrained_only)
        print(f"\n🔍 Search results for '{query}':")
    else:
        models = timm.list_models(pretrained=show_pretrained_only)
        print(f"\n📋 All models:")
    
    print(f"Found {len(models)} models\n")
    
    if len(models) == 0:
        print("❌ No models found. Try a different search term.")
        print("\nPopular categories to try:")
        print("  - vit, deit, swin, beit (transformers)")
        print("  - resnet, resnext (CNNs)")
        print("  - efficientnet, mobilenet (efficient models)")
        print("  - regnet, convnext (modern CNNs)")
        return
    
    # Show results
    if len(models) <= 20:
        for i, model in enumerate(models, 1):
            print(f"{i:3d}. {model}")
    else:
        for i, model in enumerate(models[:20], 1):
            print(f"{i:3d}. {model}")
        print(f"\n... and {len(models) - 20} more models")
        print(f"\nShowing first 20 of {len(models)} results")
    
    # Try to get info about first model
    if models:
        print(f"\n💡 Example: Creating '{models[0]}':")
        print(f"   model = timm.create_model('{models[0]}', pretrained=True)")


if __name__ == '__main__':
    print("=" * 70)
    print("TIMM MODEL SEARCH")
    print("=" * 70)
    
    # Get search term from command line or use default
    if len(sys.argv) > 1:
        query = sys.argv[1]
    else:
        query = ''
        print("\nUsage: python search_models.py <search_term>")
        print("No search term provided. Showing popular categories:\n")
        
        categories = {
            'Vision Transformers': 'vit_base',
            'DeiT': 'deit',
            'Swin': 'swin',
            'ResNet': 'resnet50',
            'EfficientNet': 'efficientnet',
            'RegNet': 'regnet',
            'ConvNeXt': 'convnext',
        }
        
        for name, pattern in categories.items():
            models = timm.list_models(f'*{pattern}*', pretrained=True)
            print(f"  {name:20s}: {len(models):3d} models (try: '{pattern}')")
        
        print("\nRun: python search_models.py <category> to see specific models")
        sys.exit(0)
    
    search_models(query, show_pretrained_only=True)
    
    print("\n" + "=" * 70)
    print("Quick Reference:")
    print("=" * 70)
    print("import timm")
    print("models = timm.list_models('*your_search*', pretrained=True)")
    print("model = timm.create_model('model_name', pretrained=True)")
    print("=" * 70)


