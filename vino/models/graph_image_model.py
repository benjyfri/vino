import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Normalize
from .dinov3_backbone import DinoV3Backbone
from .input_adapter import Conv1x1Adapter, IdentityAdapter
from .input_stems import build_input_stem
from .heads import BinaryClassificationHead, LinearClassificationHead, RegressionHead

class GraphImageModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Normalization
        if config["model"]["normalize_lvd_imagenet"]:
            self.normalize = Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        else:
            self.normalize = nn.Identity()
            
        # Adapter
        adapter_config = config["model"]["input_adapter"]
        if adapter_config["type"] == "conv1x1":
            self.adapter = Conv1x1Adapter(
                in_channels=int(adapter_config.get("in_channels", 3)),
                out_channels=int(adapter_config.get("out_channels", 3)),
            )
        elif adapter_config["type"] == "identity":
            self.adapter = IdentityAdapter()
        else:
            raise ValueError(f"Unknown input adapter type: {adapter_config['type']}")
            
        # Input stem
        stem_config = config["model"].get("input_stem", {})
        self.input_stem = build_input_stem(stem_config)
            
        # Backbone
        self.backbone = DinoV3Backbone(
            model_name=config["model"]["backbone_name"],
            freeze_mode=config["model"]["freeze_mode"],
            pretrained_path=config["model"]["pretrained_path"],
            pooling=config["model"].get("pooling", "auto"),
            revision=config["model"].get("pretrained_revision"),
        )
        
        emb_dim = self.backbone.output_dim
            
        # Head
        head_type = config["model"]["head"]["type"]
        if head_type == "binary":
            out_dim = int(config["model"]["head"].get("num_tasks", 1))
            self.head = BinaryClassificationHead(
                emb_dim, 
                config["model"]["head"]["hidden_dim"], 
                config["model"]["head"]["dropout"],
                out_dim,
            )
        elif head_type == "linear":
            out_dim = int(config["model"]["head"].get("num_tasks", 1))
            self.head = LinearClassificationHead(emb_dim, out_dim)
        elif head_type == "regression":
            self.head = RegressionHead(
                emb_dim, 
                1, 
                config["model"]["head"]["hidden_dim"], 
                config["model"]["head"]["dropout"]
            )
        else:
            raise ValueError(f"Unknown head type: {head_type}")
            
    def forward(self, images, valid_token_mask=None):
        x = self.normalize(images)
        if valid_token_mask is not None:
            input_mask = F.interpolate(valid_token_mask[:, None].float(), size=x.shape[-2:], mode="nearest")
            x = x * input_mask
        x = self.input_stem(x)
        x = self.adapter(x)
        if valid_token_mask is not None:
            mask = F.interpolate(valid_token_mask[:, None].float(), size=x.shape[-2:], mode="nearest")
            x = x * mask
        emb = self.backbone(x)
        out = self.head(emb)
        return out
