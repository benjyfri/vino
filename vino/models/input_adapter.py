import torch.nn as nn

class Conv1x1Adapter(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        return self.conv(x)
        
class IdentityAdapter(nn.Module):
    def forward(self, x):
        return x
