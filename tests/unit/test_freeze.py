from vino.models.dinov3_backbone import DinoV3Backbone

def test_freeze_mode():
    model = DinoV3Backbone(model_name="tiny", freeze_mode="frozen")
    for p in model.parameters():
        assert not p.requires_grad
