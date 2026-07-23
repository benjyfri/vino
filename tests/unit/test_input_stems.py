import torch
from vino.models.input_stems import build_input_stem, IdentityStem, ResidualCNNStem

def test_identity_stem():
    stem = IdentityStem()
    x = torch.randn(2, 3, 224, 224)
    y = stem(x)
    assert torch.all(x == y)

def test_residual_cnn_stem_shape():
    config = {
        "enabled": True,
        "type": "residual_cnn",
        "in_channels": 3,
        "hidden_channels": 32,
        "depth": 2,
        "kernel_size": 3,
        "residual_scale_init": 0.0,
        "residual_scale_trainable": True
    }
    stem = build_input_stem(config)
    assert isinstance(stem, ResidualCNNStem)
    
    # 224x224
    x = torch.randn(2, 3, 224, 224)
    y = stem(x)
    assert y.shape == x.shape
    
    # 512x512
    x2 = torch.randn(2, 3, 512, 512)
    y2 = stem(x2)
    assert y2.shape == x2.shape

def test_residual_cnn_stem_identity_init():
    config = {
        "enabled": True,
        "type": "residual_cnn",
        "residual_scale_init": 0.0
    }
    stem = build_input_stem(config)
    x = torch.randn(2, 3, 64, 64)
    y = stem(x)
    
    # With alpha=0.0, output should be exactly input
    assert torch.allclose(x, y, atol=1e-6)

def test_residual_cnn_stem_gradients():
    config = {
        "enabled": True,
        "type": "residual_cnn",
        "residual_scale_init": 0.0,
        "residual_scale_trainable": True
    }
    stem = build_input_stem(config)
    x = torch.randn(2, 3, 64, 64)
    y = stem(x)
    
    loss = y.sum()
    loss.backward()
    
    assert stem.alpha.grad is not None
    # Weights of final conv should get gradient since they multiply with x
    # Wait, if alpha is 0, gradient of alpha is non-zero, but gradient of delta parameters is scaled by alpha (which is 0).
    # So delta parameter grads might be 0. Let's just check alpha gradient and that parameters require_grad.
    assert next(stem.parameters()).requires_grad
    
    # Let's change alpha and check again
    stem.alpha.data.fill_(1.0)
    stem.zero_grad()
    y2 = stem(x)
    loss2 = y2.sum()
    loss2.backward()
    
    assert stem.final_conv.weight.grad is not None
    assert stem.final_conv.weight.grad.abs().sum() > 0

def test_build_input_stem_disabled():
    config = {
        "enabled": False,
        "type": "residual_cnn"
    }
    stem = build_input_stem(config)
    assert isinstance(stem, IdentityStem)


def _stn_cfg(t):
    return {"enabled": True, "type": t, "in_channels": 3, "hidden_channels": 16}


def test_affine_stn_is_identity_initialized():
    import torch
    from vino.models.input_stems import build_input_stem
    stn = build_input_stem(_stn_cfg("affine_stn"))
    x = torch.rand(2, 3, 64, 64)
    assert torch.allclose(stn(x), x, atol=1e-3)  # near-identity warp at init


def test_affine_stn_control_is_parameter_matched_and_non_warping():
    import torch
    from vino.models.input_stems import build_input_stem
    stn = build_input_stem(_stn_cfg("affine_stn"))
    ctrl = build_input_stem(_stn_cfg("affine_value_control"))
    assert sum(p.numel() for p in stn.parameters()) == sum(p.numel() for p in ctrl.parameters())
    # control applies a value affine (no spatial warp); at identity init a=1,b=0 -> ~identity
    x = torch.rand(2, 3, 32, 32)
    assert torch.allclose(ctrl(x), x, atol=1e-3)


def test_affine_stn_receives_gradient_and_stays_finite():
    import torch
    from vino.models.input_stems import build_input_stem
    stn = build_input_stem(_stn_cfg("affine_stn"))
    x = torch.rand(2, 3, 48, 48, requires_grad=False)
    out = stn(x)
    assert torch.isfinite(out).all()
    out.sum().backward()
    grad = sum(p.grad.abs().sum() for p in stn.parameters() if p.grad is not None)
    assert float(grad) > 0
