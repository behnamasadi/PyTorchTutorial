import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn.functional as F


# model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
# print(model.layer1)

# B, C, W, H = 1, 3, 512, 512
# x = torch.randn(B, C, W, H)

# print("stem input:", x.shape)

# x = model.conv1(x)  # 3→64
# x = model.bn1(x)
# x = model.relu(x)
# x = model.maxpool(x)
# x = model.layer1(x)

# print("stem output/ layer1 input", x.shape)

# out_layer1 = model.layer1(x)
# print("layer1 output/ layer2 input", out_layer1.shape)


# out_layer2 = model.layer2(out_layer1)
# print("layer2 output/ layer3 input", out_layer2.shape)


class ResNet18Encoder(nn.Module):
    def __init__(self, weights=ResNet18_Weights.IMAGENET1K_V1):
        super().__init__()
        resnet = resnet18(weights=weights)

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # record output channels of each feature
        self.out_channels = [64, 64, 128, 256, 512]

    def forward(self, x):
        x0 = self.conv1(x)     # [B, 64, H/2, W/2]
        x0 = self.bn1(x0)
        x0 = self.relu(x0)     # first skip
        x1 = self.maxpool(x0)  # [B, 64, H/4, W/4]
        x2 = self.layer1(x1)   # [B, 64, H/4, W/4]
        x3 = self.layer2(x2)   # [B, 128, H/8, W/8]
        x4 = self.layer3(x3)   # [B, 256, H/16, W/16]
        x5 = self.layer4(x4)   # [B, 512, H/32, W/32]
        return [x0, x2, x3, x4, x5]  # skip connections


class ConvBlock(nn.Module):
    """(Conv→BN→ReLU)×2 used for center or decoder fusion."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, up_mode="deconv"):
        super().__init__()
        if up_mode == "deconv":
            self.up = nn.ConvTranspose2d(
                in_ch, out_ch, kernel_size=2, stride=2)
        elif up_mode == "bilinear":
            self.up = nn.Identity()
            self.up_out_ch = in_ch  # keep channels, reduce after concat in ConvBlock
        else:
            raise ValueError("up_mode must be 'deconv' or 'bilinear'")

        # If using bilinear upsampling, we first upsample then reduce channels via 1x1
        self.reduce = nn.Identity() if up_mode == "deconv" else nn.Conv2d(
            in_ch, out_ch, 1, bias=False)
        self.fuse = ConvBlock(out_ch + skip_ch, out_ch)

        self.up_mode = up_mode

    def forward(self, x, skip):
        if self.up_mode == "deconv":
            x = self.up(x)
        else:
            x = F.interpolate(
                x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = self.reduce(x)

        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(
                x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.fuse(x)


class ResNetUNet(nn.Module):
    def __init__(self, num_classes=1, encoder_weights=ResNet18_Weights.IMAGENET1K_V1, center_mult=1.0, up_mode="deconv"
                 ):
        """
        center_mult lets you widen/narrow the center block:
          center_out_ch = int(enc_chs[-1] * center_mult)
        """
        super().__init__()

        self.encoder = ResNet18Encoder(weights=encoder_weights)

        enc_chs = self.encoder.out_channels  # [64, 64, 128, 256, 512]
        c0, c2, c3, c4, c5 = enc_chs

        # Optional explicit bottleneck (center) on top of x5
        center_out = int(c5 * center_mult)
        self.center = ConvBlock(
            c5, center_out) if center_mult != 1.0 else nn.Identity()
        bottom_ch = center_out if center_mult != 1.0 else c5

        # Decoder: parameterized, no hardcoded 512
        self.dec4 = DecoderBlock(bottom_ch, c4, c4, up_mode=up_mode)  # 32→16
        self.dec3 = DecoderBlock(c4,       c3, c3, up_mode=up_mode)   # 16→8
        self.dec2 = DecoderBlock(c3,       c2, c2, up_mode=up_mode)   # 8→4
        self.dec1 = DecoderBlock(c2,       c0, c0, up_mode=up_mode)   # 4→2

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear",
                        align_corners=False),  # 2→1
            nn.Conv2d(c0, num_classes, kernel_size=1),
        )

    def forward(self, x):
        x0, x2, x3, x4, x5 = self.encoder(x)
        x5 = self.center(x5)        # explicit U-Net bottleneck (optional)
        d4 = self.dec4(x5, x4)
        d3 = self.dec3(d4, x3)
        d2 = self.dec2(d3, x2)
        d1 = self.dec1(d2, x0)
        return self.final(d1)


if __name__ == "__main__":
    model = ResNetUNet(num_classes=10)
    # x = torch.randn(1, 3, 224, 224)
    x = torch.randn(1, 3, 512, 512)
    y = model(x)
    print(y.shape)  # → [1, num_classes, 224, 224]
