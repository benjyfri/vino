import torch
import torch.nn as nn
from transformers import AutoModel

class DinoV3Backbone(nn.Module):
    def __init__(self, model_name: str = "facebook/dinov3-vits16-pretrain-lvd1689m",
                 freeze_mode: str = "frozen", pretrained_path: str = None,
                 pooling: str = "auto", revision: str | None = None):
        super().__init__()
        # Tiny dummy model for tests
        if "tiny" in model_name and "dinov3" not in model_name:
            self.model = TinyBackbone()
        else:
            path = pretrained_path if pretrained_path else model_name
            try:
                self.model = AutoModel.from_pretrained(path, revision=revision)
            except Exception as exc:
                raise RuntimeError(f"Failed to load requested pretrained backbone {path!r}") from exc
        self.output_dim = self._output_dim()
        self.pooling = pooling
                
        self.apply_freeze(freeze_mode)
        
    def forward(self, pixel_values):
        kwargs = {"pixel_values": pixel_values}
        if not isinstance(self.model, TinyBackbone):
            kwargs["interpolate_pos_encoding"] = True
            
        try:
            outputs = self.model(**kwargs)
        except TypeError:
            outputs = self.model(pixel_values=pixel_values)
        if self.pooling == "auto" and hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            return outputs.pooler_output
        elif hasattr(outputs, "last_hidden_state"):
            hidden = outputs.last_hidden_state
            if hidden.ndim == 3:
                if self.pooling == "mean":
                    return hidden.mean(dim=1)
                return hidden[:, 0]
            return hidden.mean(dim=tuple(range(2, hidden.ndim)))
        else:
            return outputs[0][:, 0]

    def _output_dim(self) -> int:
        if isinstance(self.model, TinyBackbone):
            return 16
        config = self.model.config
        if getattr(config, "hidden_size", None):
            return int(config.hidden_size)
        if getattr(config, "hidden_sizes", None):
            return int(config.hidden_sizes[-1])
        raise ValueError(f"Cannot infer output dimension from {type(config).__name__}")
            
    def apply_freeze(self, mode: str):
        if mode == "full":
            return
        elif mode == "frozen":
            freeze_all(self.model)
        elif mode == "last2":
            freeze_all(self.model)
            # Try to unfreeze last two blocks if it's a ViT
            if hasattr(self.model, "encoder") and hasattr(self.model.encoder, "layer"):
                layers = self.model.encoder.layer
                for layer in layers[-2:]:
                    unfreeze_module(layer)
        elif mode == "last1":
            freeze_all(self.model)
            if hasattr(self.model, "encoder") and hasattr(self.model.encoder, "layer"):
                layers = self.model.encoder.layer
                unfreeze_module(layers[-1])
        elif mode in ["train_patch_embed", "train_stem_and_patch_embed"]:
            freeze_all(self.model)
            set_trainable_patch_embedding(self.model, mode_name=mode)
        else:
            raise ValueError(f"Unknown freeze mode: {mode}")
            
def freeze_all(module: nn.Module):
    for param in module.parameters():
        param.requires_grad = False

def unfreeze_module(module: nn.Module):
    for param in module.parameters():
        param.requires_grad = True

def set_trainable_patch_embedding(model: nn.Module, mode_name: str):
    patch_embed_module = None
    
    # Try different known paths
    if hasattr(model, "embeddings") and hasattr(model.embeddings, "patch_embeddings"):
        patch_embed_module = model.embeddings.patch_embeddings
    elif hasattr(model, "vision_model") and hasattr(model.vision_model, "embeddings") and hasattr(model.vision_model.embeddings, "patch_embedding"):
        patch_embed_module = model.vision_model.embeddings.patch_embedding
    elif hasattr(model, "patch_embed"):
        patch_embed_module = model.patch_embed
    elif hasattr(model, "conv"): # for TinyBackbone
        patch_embed_module = model.conv
        
    if patch_embed_module is None:
        # try searching for first Conv2d with stride > 1 and kernel_size > 1
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                patch_embed_module = module
                break
                
    if patch_embed_module is None:
        raise ValueError(f"Failed to locate patch embedding module in {model.__class__.__name__} for requested freeze mode '{mode_name}'. Cannot apply train_patch_embed.")
        
    unfreeze_module(patch_embed_module)
    unfrozen_params = sum(p.numel() for p in patch_embed_module.parameters() if p.requires_grad)
    print(f"[debug] Unfrozen patch embedding module ({patch_embed_module.__class__.__name__}) with {unfrozen_params} trainable params.", flush=True)

class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 16, stride=16)
        
    def forward(self, pixel_values=None, **kwargs):
        x = self.conv(pixel_values)
        b, c, h, w = x.shape
        x = x.view(b, c, -1).mean(dim=-1) # average pool
        class DummyOutput:
            pooler_output = x
        return DummyOutput()
