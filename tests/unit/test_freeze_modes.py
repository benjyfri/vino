import pytest
import torch
import torch.nn as nn
from vino.models.dinov3_backbone import DinoV3Backbone

class DummyViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = type('obj', (object,), {'patch_embeddings': nn.Conv2d(3, 384, kernel_size=16, stride=16)})()
        self.encoder = type('obj', (object,), {'layer': nn.ModuleList([nn.Linear(384, 384) for _ in range(2)])})()
        self.pooler = nn.Linear(384, 384)

    def parameters(self):
        for m in [self.embeddings.patch_embeddings, self.encoder.layer, self.pooler]:
            for p in m.parameters():
                yield p

    def named_modules(self):
        yield "embeddings.patch_embeddings", self.embeddings.patch_embeddings
        yield "encoder.layer", self.encoder.layer
        yield "pooler", self.pooler

def test_freeze_frozen():
    backbone = DinoV3Backbone(model_name="tiny", freeze_mode="frozen")
    for p in backbone.model.parameters():
        assert not p.requires_grad

def test_train_patch_embed():
    # We must patch DinoV3Backbone to use our DummyViT to test the logic
    # because TinyBackbone has no blocks
    backbone = DinoV3Backbone(model_name="tiny", freeze_mode="full")
    backbone.model = DummyViT()
    
    # Test frozen
    backbone.apply_freeze("frozen")
    for p in backbone.model.parameters():
        assert not p.requires_grad
        
    # Test train_patch_embed
    backbone.apply_freeze("train_patch_embed")
    
    # patch embeddings should be unfreezed
    assert next(backbone.model.embeddings.patch_embeddings.parameters()).requires_grad
    
    # blocks should be frozen
    assert not next(backbone.model.encoder.layer[0].parameters()).requires_grad
    assert not next(backbone.model.pooler.parameters()).requires_grad

def test_missing_patch_embed_fails():
    backbone = DinoV3Backbone(model_name="tiny", freeze_mode="full")
    # Make a dummy model with no conv/patch_embed
    class BadModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 10)
    backbone.model = BadModel()
    
    with pytest.raises(ValueError, match="Failed to locate patch embedding module"):
        backbone.apply_freeze("train_patch_embed")
