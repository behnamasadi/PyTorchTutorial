import torch
import torch.nn as nn
import torch.nn.functional as F

# SSIM loss function
class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average

    def gaussian_window(self, window_size, sigma):
        gauss = torch.Tensor([torch.exp(-(x - window_size//2)**2/float(2*sigma**2)) 
                              for x in range(window_size)])
        return gauss/gauss.sum()

    def create_window(self, channel):
        _1D_window = self.gaussian_window(self.window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, self.window_size, self.window_size).contiguous()
        return window

    def forward(self, img1, img2):
        channel = img1.size(1)
        window = self.create_window(channel).to(img1.device)

        mu1 = F.conv2d(img1, window, padding=self.window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size//2, groups=channel)

        mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2
        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size//2, groups=channel) - mu1_mu2

        C1, C2 = 0.01**2, 0.03**2
        ssim_map = ((2*mu1_mu2 + C1) * (2*sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if self.size_average:
            return 1 - ssim_map.mean()  # as loss
        else:
            return 1 - ssim_map.mean(1).mean(1).mean(1)

# Example usage
loss_fn = SSIMLoss()
img1 = torch.rand((1, 3, 128, 128))
img2 = torch.rand((1, 3, 128, 128))

loss = loss_fn(img1, img2)
print("SSIM Loss:", loss.item())
