
# Modified Code for Flexible Resolution:


class UNet(nn.Module):
    def __init__(self, base=64, num_classes=19, input_size=(256, 256)):
        super(UNet, self).__init__()
        self.base = base
        self.input_size = input_size
        self.num_classes = num_classes
        
        # -------- Encoder (unchanged) --------
        self.E1conv1 = torch.nn.Conv2d(3, base, kernel_size=3, stride=1, padding=1, bias=False)
        # ... rest of encoder unchanged ...
        
        # -------- Decoder (unchanged) --------
        self.D4upsample = torch.nn.ConvTranspose2d(base*16, base*8, kernel_size=2, stride=2)
        # ... rest of decoder unchanged ...
        
    def forward(self, x):
        # Verify input size (optional)
        if x.shape[-2:] != self.input_size:
            print(f"Warning: Expected {self.input_size}, got {x.shape[-2:]}")
        
        # Forward pass (unchanged)
        # E1
        e1 = self.E1conv2(self.E1conv1(x))
        e1_pool = self.pool1(e1)
        
        # E2  
        e2 = self.E2conv2(self.E2conv1(e1_pool))
        e2_pool = self.pool2(e2)
        
        # ... rest unchanged ...
        
        # Decoder with skip connections
        d4_up = self.D4upsample(bottleneck)
        d4_concat = torch.cat([d4_up, e4], dim=1)  # Skip connection
        d4 = self.D4conv2(self.D4conv1(d4_concat))
        # ... rest unchanged ...
        
        return self.final_conv(d1)
