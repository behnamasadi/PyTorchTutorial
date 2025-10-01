# block_cls → a class (e.g., BasicBlock)
# block_mod → an instance of that class (a module you add to the network)

from typing import List, Sequence, Type
import torch
import torch.nn as nn


# ---------- Base interface for residual blocks ----------
class ResidualBlockBase(nn.Module):
    """
    Base class to type residual blocks.
    Child classes must define a class attribute `expansion` (int).
    """
    expansion: int = 1  # how many times `planes` the block outputs (Basic=1, Bottleneck=4)


# ---------- Basic residual block (ResNet-18/34 style) ----------
class BasicBlock(ResidualBlockBase):
    expansion: int = 1

    def __init__(self, in_channels: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        out_channels = planes * self.expansion

        self.conv1 = nn.Conv2d(in_channels, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Identity or 1x1 projection on the skip path to match shape
        needs_projection = (stride != 1) or (in_channels != out_channels)
        if needs_projection:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)  # <-- "skip connection" path

        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)

        y = self.conv2(y)
        y = self.bn2(y)

        y = y + identity          # residual addition: F(x) + x
        y = self.relu(y)
        return y


# ---------- ResNet backbone using a block CLASS (not an instance) ----------
class ResNet(nn.Module):
    """
    Generic ResNet that can be built with different block classes (e.g., BasicBlock, Bottleneck).

    Parameters
    ----------
    block_cls : Type[ResidualBlockBase]
        The CLASS of the residual block to instantiate (e.g., `BasicBlock`), not an instance.
    layers_per_stage : Sequence[int]
        Number of residual blocks in each of the 4 stages, e.g. [2,2,2,2] or [3,4,6,3].
    num_classes : int
        Output classes for the final classifier.
    in_channels : int
        Input image channels (3 for RGB).
    """

    def __init__(
        self,
        block_cls: Type[ResidualBlockBase],
        layers_per_stage: Sequence[int],
        num_classes: int = 1000,
        in_channels: int = 3,
    ) -> None:
        super().__init__()

        assert len(layers_per_stage) == 4, "Expected 4 stages"
        self.block_cls: Type[ResidualBlockBase] = block_cls

        # ----- Stem -----
        self.stem_conv = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.stem_bn = nn.BatchNorm2d(64)
        self.stem_relu = nn.ReLU(inplace=True)
        self.stem_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Track current #channels flowing into the next stage
        self.current_channels: int = 64

        # ----- Stages (conv2_x .. conv5_x) -----

        print("# ----- Stage 1 -----")
        self.layer1 = self._build_stage(
            planes=64,  num_blocks=layers_per_stage[0], first_stride=1)

        print("# ----- Stage 2 -----")
        self.layer2 = self._build_stage(
            planes=128, num_blocks=layers_per_stage[1], first_stride=2)

        print("# ----- Stage 3 -----")
        self.layer3 = self._build_stage(
            planes=256, num_blocks=layers_per_stage[2], first_stride=2)

        print("# ----- Stage 4 -----")
        self.layer4 = self._build_stage(
            planes=512, num_blocks=layers_per_stage[3], first_stride=2)

        # ----- Head -----
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * self.block_cls.expansion, num_classes)

        # He init for convs; BN to ones/zeros (standard for ResNet)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def _build_stage(self, planes: int, num_blocks: int, first_stride: int) -> nn.Sequential:
        """
        Build one ResNet stage (e.g., conv3_x), returning a Sequential of residual block INSTANCES.

        - The first block may downsample via `first_stride`.
        - Subsequent blocks keep stride=1.
        """
        print(
            f"planes: {planes}, num_blocks: {num_blocks}, first_stride: {first_stride}")

        blocks: List[nn.Module] = []

        # 1) First block in the stage: may change spatial size & width
        print(
            f"First block in the stage, in_channels={self.current_channels}, planes={planes}, stride={first_stride}")
        block_mod = self.block_cls(  # <-- instantiate the CLASS
            in_channels=self.current_channels,
            planes=planes,
            stride=first_stride,
        )
        blocks.append(block_mod)

        # After the first block, the stage's channel width is planes * expansion
        self.current_channels = planes * self.block_cls.expansion
        print(
            f"After the first block, the stage's channel width is planes * expansion={self.current_channels}")

        # 2) Remaining blocks: keep same width and stride=1
        for i in range(1, num_blocks):
            block_mod = self.block_cls(
                in_channels=self.current_channels,
                planes=planes,
                stride=1,
            )
            blocks.append(block_mod)

            print(
                f"block number {i}, in_channels={self.current_channels}, planes={planes}, stride=1")

        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stem
        x = self.stem_conv(x)
        x = self.stem_bn(x)
        x = self.stem_relu(x)
        x = self.stem_pool(x)

        # Stages
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Head
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# ---------- Factory helpers ----------
def resnet18(num_classes: int = 1000, in_channels: int = 3) -> ResNet:
    """ResNet-18 uses BasicBlock with layers_per_stage = [2, 2, 2, 2]."""
    print(
        f"ResNet-18 uses BasicBlock with layers_per_stage = [2, 2, 2, 2], and in_channels={in_channels}")
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, in_channels=in_channels)


def resnet34(num_classes: int = 1000, in_channels: int = 3) -> ResNet:
    """ResNet-34 uses BasicBlock with layers_per_stage = [3, 4, 6, 3]."""
    print(
        f"ResNet-34 uses BasicBlock with layers_per_stage = [3, 4, 6, 3]., and in_channels= {in_channels}")
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, in_channels=in_channels)


# ---------- Quick smoke test ----------
if __name__ == "__main__":
    # model = resnet18(num_classes=10)
    # x = torch.randn(2, 3, 224, 224)
    # logits = model(x)
    # print(type(BasicBlock))         # <class 'type'>  (a CLASS)
    # # True (an INSTANCE inside the stage)
    # print(isinstance(model.layer1[0], BasicBlock))
    # print(logits.shape)             # torch.Size([2, 10])

    # ------------------------------------

    model = resnet34(num_classes=10)
    x = torch.randn(2, 3, 224, 224)
    logits = model(x)
    print(type(BasicBlock))         # <class 'type'>  (a CLASS)
    # True (an INSTANCE inside the stage)
    print(isinstance(model.layer1[0], BasicBlock))
    print(logits.shape)             # torch.Size([2, 10])
