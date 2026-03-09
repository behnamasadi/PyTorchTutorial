from monai.networks.nets import UNETR
import monai
import torch
from monai.networks.nets import UNet
import inspect

print(monai.__version__)
print(torch.cuda.is_available())


# for k, v in monai.networks.nets.__dict__.items():
#     print(k)


# model = UNETR(
#     in_channels=1,
#     out_channels=14,
#     img_size=(96, 96, 96),
#     feature_size=16,
#     hidden_size=768,
#     mlp_dim=3072,
#     num_heads=12,
#     spatial_dims=3,
#     pretrained=True,
#     pretrained_name="UNETR_BTCV"
# )
