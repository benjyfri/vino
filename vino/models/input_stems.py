import torch
import torch.nn as nn
import torch.nn.functional as F

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

class AffineSpatialTransformer(nn.Module):
    """Identity-initialized affine spatial transformer (learned 2x3 warp) with safeguards.

    A small localization net predicts a bounded deviation from the identity affine; translation and
    scale/shear deviations are squashed with tanh to a configured range so the grid cannot collapse
    or crop to nothing. Because graph-image coordinates encode canonical structure, warping may
    damage semantics -- always compare against :class:`AffineValueControl` (same parameter count,
    no spatial warp).
    """

    def __init__(self, config):
        super().__init__()
        in_ch = int(config.get("in_channels", 3))
        hidden = int(config.get("hidden_channels", 16))
        self.max_translate = float(config.get("max_translate", 0.3))
        self.max_delta = float(config.get("max_scale_shear_delta", 0.3))
        self.warp = bool(config.get("warp", True))
        self.loc = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d(4), nn.Flatten(),
            nn.Linear(hidden * 16, 32), nn.GELU(),
        )
        self.theta_head = nn.Linear(32, 6)
        nn.init.zeros_(self.theta_head.weight)
        self.theta_head.bias.data.copy_(torch.tensor([1.0, 0, 0, 0, 1.0, 0]))

    def _theta(self, x):
        raw = self.theta_head(self.loc(x)).view(-1, 2, 3)
        base = torch.tensor([1.0, 0, 0, 0, 1.0, 0], device=x.device, dtype=x.dtype).view(1, 2, 3)
        delta = raw - base
        # bound scale/shear (the 2x2 block) and translation (last column) separately
        lin = torch.tanh(delta[:, :, :2]) * self.max_delta
        trans = torch.tanh(delta[:, :, 2:]) * self.max_translate
        return base + torch.cat([lin, trans], dim=2)

    def forward(self, x):
        theta = self._theta(x)
        if not self.warp:
            return x
        grid = F.affine_grid(theta, list(x.shape), align_corners=False)
        det = theta[:, 0, 0] * theta[:, 1, 1] - theta[:, 0, 1] * theta[:, 1, 0]
        if torch.any(det.abs() < 1e-3):
            raise FloatingPointError("AffineSpatialTransformer produced a near-degenerate grid")
        out = F.grid_sample(x, grid, align_corners=False, padding_mode="zeros")
        if not torch.isfinite(out).all():
            raise FloatingPointError("AffineSpatialTransformer produced non-finite output")
        return out


class AffineValueControl(AffineSpatialTransformer):
    """Parameter-matched control for the STN: identical localization net + 6-value head, but the
    predicted parameters are applied as a per-image value affine (x*a + b) with NO spatial warp,
    isolating whether any STN gain comes from warping or merely from the extra parameters."""

    def forward(self, x):
        theta = self._theta(x)  # [B,2,3]
        a = theta[:, 0, 0].view(-1, 1, 1, 1)
        b = theta[:, 0, 2].view(-1, 1, 1, 1)
        return (x * a + b).clamp(0.0, 1.0)


def build_input_stem(config):
    if not config:
        return IdentityStem()

    enabled = config.get("enabled", False)
    if not enabled:
        return IdentityStem()

    stem_type = config.get("type", "residual_cnn")
    if stem_type in ["residual_cnn", "ResidualCNNStem", "residual"]:
        return ResidualCNNStem(config)
    if stem_type in ("affine_stn", "affine_spatial_transformer"):
        return AffineSpatialTransformer(config)
    if stem_type in ("affine_value_control", "affine_stn_control"):
        return AffineValueControl(config)
    raise ValueError(f"Unknown input stem type: {stem_type}")
