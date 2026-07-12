import torch
import torch.nn as nn

class IdentityStem(nn.Module):
    def forward(self, x):
        return x

class ResidualCNNStem(nn.Module):
    def __init__(self, config):
        super().__init__()
        in_channels = config.get("in_channels", 3)
        hidden_channels = config.get("hidden_channels", 32)
        depth = config.get("depth", 2)
        kernel_size = config.get("kernel_size", 3)
        norm_type = config.get("norm", "none")
        activation_type = config.get("activation", "gelu")
        self.residual_scale_init = config.get("residual_scale_init", 0.0)
        residual_scale_trainable = config.get("residual_scale_trainable", True)
        dropout_p = config.get("dropout", 0.0)

        padding = kernel_size // 2

        layers = []
        layers.append(nn.Conv2d(in_channels, hidden_channels, kernel_size, padding=padding))
        
        for _ in range(depth - 1):
            if norm_type == "batchnorm":
                layers.append(nn.BatchNorm2d(hidden_channels))
            elif norm_type == "groupnorm":
                layers.append(nn.GroupNorm(min(hidden_channels, 8), hidden_channels))
                
            if activation_type == "gelu":
                layers.append(nn.GELU())
            elif activation_type == "relu":
                layers.append(nn.ReLU())
            elif activation_type == "silu":
                layers.append(nn.SiLU())
                
            if dropout_p > 0.0:
                layers.append(nn.Dropout2d(p=dropout_p))
                
            layers.append(nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=padding))
            
        if activation_type == "gelu":
            layers.append(nn.GELU())
        elif activation_type == "relu":
            layers.append(nn.ReLU())
        elif activation_type == "silu":
            layers.append(nn.SiLU())

        self.body = nn.Sequential(*layers)
        self.final_conv = nn.Conv2d(hidden_channels, in_channels, 1)

        if residual_scale_trainable:
            self.alpha = nn.Parameter(torch.tensor(float(self.residual_scale_init)))
        else:
            self.register_buffer("alpha", torch.tensor(float(self.residual_scale_init)))

    def forward(self, x):
        h = self.body(x)
        delta = self.final_conv(h)
        return x + self.alpha * delta

def build_input_stem(config):
    if not config:
        return IdentityStem()
    
    enabled = config.get("enabled", False)
    if not enabled:
        return IdentityStem()
        
    stem_type = config.get("type", "residual_cnn")
    if stem_type in ["residual_cnn", "ResidualCNNStem", "residual"]:
        return ResidualCNNStem(config)
    else:
        raise ValueError(f"Unknown input stem type: {stem_type}")
