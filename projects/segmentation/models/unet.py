import torch
import torch.nn as nn

from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models.segmentation import DeepLabV3
# The following semantic segmentation models are available, with or without pre-trained weights:

# DeepLabV3
# FCN
# LRASPP


weight = ResNet18_Weights.IMAGENET1K_V1
model = resnet18(weights=weight)


class UNet(torch.nn.Module):
    def __init__(self, input_channel=3, n_classes=1, base=64):
        super().__init__()

        self.relu = nn.ReLU(inplace=True)

        # -------- Encoder --------

        # -------- E1 --------
        # E1 in_channels is: 256 × 256 x 3, out_channels after max-pooling 128 x 128 x (base = 64)
        self.E1conv1 = torch.nn.Conv2d(
            # channels: 3 -> base
            in_channels=input_channel, out_channels=base, stride=1, padding=1, kernel_size=3, bias=False)

        self.E1conv2 = torch.nn.Conv2d(
            # channels: base -> base
            in_channels=base, out_channels=base, stride=1, padding=1, kernel_size=3, bias=False)

        self.pool1 = torch.nn.MaxPool2d(
            kernel_size=2, stride=2)  # W,H 256 × 256 ->128 x 128

        # -------- E2 --------
        # E2 in_channels is: 128 x 128 x base, out_channels after max-pooling  64 x 64 x (base*2 = 128)
        self.E2conv1 = torch.nn.Conv2d(
            # channels: base -> base*2
            in_channels=base, out_channels=base*2, stride=1, padding=1, kernel_size=3, bias=False)

        self.E2conv2 = torch.nn.Conv2d(
            # channels: base -> base*2
            in_channels=base*2, out_channels=base*2, stride=1, padding=1, kernel_size=3, bias=False)

        self.pool2 = torch.nn.MaxPool2d(
            kernel_size=2, stride=2)  # W,H 128 x 128 -> 64 x 64

        # -------- E3 --------
        # E3 in_channels is: 64 x 64 x base*2 = 128, out_channels after max-pooling 32 x 32 x (base*4)
        self.E3conv1 = torch.nn.Conv2d(
            # channels: base*2 -> base*4
            in_channels=base*2, out_channels=base*4, stride=1, padding=1, kernel_size=3, bias=False)

        self.E3conv2 = torch.nn.Conv2d(
            # channels: base*4 -> base*4
            in_channels=base*4, out_channels=base*4, stride=1, padding=1, kernel_size=3, bias=False)

        self.pool3 = torch.nn.MaxPool2d(
            kernel_size=2, stride=2)  # W,H 64 x 64 -> 32 x 32

        # -------- E4 --------
        # E4 in_channels is: 32 x 32x base*4=256, out_channels after max-pooling 16 x 16 x (base*8=512)
        self.E4conv1 = torch.nn.Conv2d(
            # channels: base*4 -> base*8
            in_channels=base*4, out_channels=base*8, stride=1, padding=1, kernel_size=3, bias=False)

        self.E4conv2 = torch.nn.Conv2d(
            # channels: base*8 -> base*8
            in_channels=base*8, out_channels=base*8, stride=1, padding=1, kernel_size=3, bias=False)

        self.pool4 = torch.nn.MaxPool2d(
            kernel_size=2, stride=2)  # W,H  32 x 32 -> 16 x 16

        # -------- Bottleneck --------

        # Bottleneck in_channels is: 16 x 16 x base*8=512, out_channels is 16 x 16 x (base*16=1024)
        self.Bottleneckconv1 = torch.nn.Conv2d(
            # channels: base*8 -> base*16
            in_channels=base*8, out_channels=base*16, stride=1, padding=1, kernel_size=3, bias=False)

        self.Bottleneckconv2 = torch.nn.Conv2d(
            # channels: base*16 -> base*16
            in_channels=base*16, out_channels=base*16, stride=1, padding=1, kernel_size=3, bias=False)

        # -------- Decoder --------

        # -------- D4 --------
        # First we upsample the input which is coming from Bottleneck, which is 16 x 16 x (base*16=1024)
        # After upsampling we have 32 x 32 x base*8, so we reduce the number of channels by half and make each feature map 4 times bigger
        # Then we concatenate E4 which is 32 x 32 x base*8 (we get E4 before max pooling so it is still 32 x 32),
        # Therefore in D4conv1 we have  in_channels= base*8 + base*8 because half of the channels are coming from E4 and other half from upsampling
        self.D4upsample = torch.nn.ConvTranspose2d(
            # channels: base*16 -> base*8
            kernel_size=2, stride=2, in_channels=base*16, out_channels=base*8)

        self.D4conv1 = torch.nn.Conv2d(
            # In base*8 + base*8 first base*8 comes from upsampling an second one E4
            # channels: base*16 -> base*8 (1024 -> 512)
            in_channels=base*8 + base*8, out_channels=base*8, kernel_size=3, padding=1, bias=False)

        self.D4conv2 = torch.nn.Conv2d(
            # channels: base*8 -> base*8
            in_channels=base*8, out_channels=base*8, kernel_size=3, padding=1, bias=False)

        # -------- D3 --------
        # First we upsample the input which is coming from D4, which is 32 x 32 x (base*8=512)
        # After upsampling we have 64 x 64 x base*4, so we reduce the number of channels by half and make each feature map 4 times bigger
        # Then we concatenate E3 which is 64 x 64 x base*4 (we get E3 before max pooling so it is still 64 x 64),
        # Therefore in D3conv1 we have  in_channels= base*4 + base*4 because half of the channels are coming from E3 and other half from upsampling
        self.D3upsample = torch.nn.ConvTranspose2d(
            # channels: base*8 -> base*4
            kernel_size=2, stride=2, in_channels=base*8, out_channels=base*4)

        self.D3conv1 = torch.nn.Conv2d(
            # In base*4 + base*4 first base*4 comes from upsampling an second one E3
            # channels: base*8 -> base*4 (512 -> 256)
            in_channels=base*4 + base*4, out_channels=base*4, kernel_size=3, padding=1, bias=False)

        self.D3conv2 = torch.nn.Conv2d(
            # channels: base*4 -> base*4
            in_channels=base*4, out_channels=base*4, kernel_size=3, padding=1, bias=False)

        # -------- D2 --------
        # First we upsample the input which is coming from D3, which is 64 x 64 x (base*4=256)
        # After upsampling we have 128 x 128 x base*2, so we reduce the number of channels by half and make each feature map 4 times bigger
        # Then we concatenate E2 which is 128 x 128 x base*2 (we get E2 before max pooling so it is still 128 x 128),
        # Therefore in D2conv1 we have  in_channels= base*2 + base*2 because half of the channels are coming from E2 and other half from upsampling
        self.D2upsample = torch.nn.ConvTranspose2d(
            # channels: base*4 -> base*2
            kernel_size=2, stride=2, in_channels=base*4, out_channels=base*2)

        self.D2conv1 = torch.nn.Conv2d(
            # In base*2 + base*2 first base*2 comes from upsampling an second one E2
            # channels: base*4 -> base*2 (256 -> 128)
            in_channels=base*2 + base*2, out_channels=base*2, kernel_size=3, padding=1, bias=False)

        self.D2conv2 = torch.nn.Conv2d(
            # channels: base*2 -> base*2
            in_channels=base*2, out_channels=base*2, kernel_size=3, padding=1, bias=False)

        # -------- D1 --------
        # First we upsample the input which is coming from D2, which is 128 x 128 x (base*2=128)
        # After upsampling we have 256 x 256 x base, so we reduce the number of channels by half and make each feature map 4 times bigger
        # Then we concatenate E1 which is 256 x 256 x base (we get E1 before max pooling so it is still 256 x 256),
        # Therefore in D1conv1 we have  in_channels= base + base because half of the channels are coming from E1 and other half from upsampling
        self.D1upsample = torch.nn.ConvTranspose2d(
            # channels: base*2 -> base
            kernel_size=2, stride=2, in_channels=base*2, out_channels=base)

        self.D1conv1 = torch.nn.Conv2d(
            # In base + base first base comes from upsampling an second one E1
            # channels: base*2 -> base (128 -> 64)
            in_channels=base + base, out_channels=base, kernel_size=3, padding=1, bias=False)

        self.D1conv2 = torch.nn.Conv2d(
            # channels: base -> base
            in_channels=base, out_channels=base, kernel_size=3, padding=1, bias=False)

        # -------- Final Output --------
        # Finally we convert the feature maps to the desired number of segmentation classes
        # Input is 256 x 256 x base (64), output is 256 x 256 x num_classes
        self.final_conv = torch.nn.Conv2d(
            # channels: base -> num_classes
            in_channels=base, out_channels=n_classes, kernel_size=1)

    def forward(self, x):
        E1 = self.E1conv1(x)
        E1 = self.relu(self.E1conv2(E1))  # [B, base, H/2, W/2]
        x = self.pool1(E1)  # [B, base, H/2, W/2]

        E2 = self.E2conv1(x)
        E2 = self.relu(self.E2conv2(E2))
        x = self.pool2(E2)

        E3 = self.E3conv1(x)
        E3 = self.relu(self.E3conv2(E3))
        x = self.pool3(E3)

        E4 = self.E4conv1(x)
        E4 = self.relu(self.E4conv2(E4))
        x = self.pool4(E4)

        Bottleneck = self.Bottleneckconv1(x)
        Bottleneck = self.relu(self.Bottleneckconv2(Bottleneck))

        # D4
        # upsample bottleneck (base*16 -> base*8)
        x = self.D4upsample(Bottleneck)
        x = torch.cat([x, E4], dim=1)    # concat with E4
        x = self.relu(self.D4conv1(x))
        D4 = self.relu(self.D4conv2(x))

        # D3
        x = self.D3upsample(D4)
        x = torch.cat([x, E3], dim=1)
        x = self.relu(self.D3conv1(x))
        D3 = self.relu(self.D3conv2(x))

        # D2
        x = self.D2upsample(D3)
        x = torch.cat([x, E2], dim=1)
        x = self.relu(self.D2conv1(x))
        D2 = self.relu(self.D2conv2(x))

        # D1
        x = self.D1upsample(D2)
        x = torch.cat([x, E1], dim=1)
        x = self.relu(self.D1conv1(x))
        D1 = self.relu(self.D1conv2(x))

        return self.final_conv(D1)


if __name__ == "__main__":

    # a = torch.randn([1, 4, 3, 2])
    # b = torch.randn([1, 1, 3, 2])
    # c = torch.cat([a, b], dim=1)
    # print(c)
    # print(c.shape)

    model = UNet(input_channel=3, n_classes=1, base=64)

    # Create a sample input (batch_size=1, channels=3, height=256, width=256)
    x = torch.randn(1, 3, 256, 256)

    # Forward pass
    output = model(x)

    # Check output shape
    print(output.shape)  # Should be [1, 1, 256, 256] for n_classes=1
