import torch
import torch.nn


class SimpleNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(
            in_channels=3, out_channels=4, kernel_size=3, padding=1, stride=2, bias=False)

        self.FC = torch.nn.Sequential(
            torch.nn.Linear(in_features=4*28*28, out_features=1000),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=1000, out_features=100)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = x.flatten(1)
        x = self.FC(x)
        return x


model = SimpleNet()


B, C, H, W = 1, 3, 56, 56
x = torch.randn(B, C, H, W)

out = model(x)
print(out.shape)


for name, param in model.named_parameters():
    print(name, param.shape)
