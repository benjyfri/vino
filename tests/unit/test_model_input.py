import torch
from vino.transforms.model_input import transform_graph_image_for_model

def test_resize_bilinear():
    image = torch.ones((3, 256, 256)) * 0.5
    valid_node_mask = torch.zeros(256, dtype=torch.bool)
    valid_node_mask[:15] = True
    
    transformed = transform_graph_image_for_model(
        image=image,
        valid_node_mask=valid_node_mask,
        input_mode="resize_bilinear",
        input_size=224
    )
    
    assert transformed["image"].shape == (3, 224, 224)
    assert transformed["n_valid"] == 15
    assert transformed["valid_token_mask"] is None
    assert torch.all(transformed["image"] >= 0.0) and torch.all(transformed["image"] <= 1.0)
    assert not torch.all(transformed["image"] == 0.0)

def test_patch_aligned_repeat():
    image = torch.zeros((3, 8, 8))
    # Fill crop with 0.5
    image[:, :4, :4] = 0.5
    
    valid_node_mask = torch.zeros(8, dtype=torch.bool)
    valid_node_mask[:4] = True
    
    transformed = transform_graph_image_for_model(
        image=image,
        valid_node_mask=valid_node_mask,
        input_mode="patch_aligned_repeat",
        patch_size=2,
        token_grid_size=6
    )
    
    out_img = transformed["image"]
    assert out_img.shape == (3, 12, 12)
    assert transformed["n_valid"] == 4
    
    # Check valid token mask
    vmask = transformed["valid_token_mask"]
    assert vmask.shape == (6, 6)
    assert torch.all(vmask[:4, :4] == True)
    assert torch.all(vmask[4:, :] == False)
    assert torch.all(vmask[:, 4:] == False)
    
    # Check that padded region is 0
    assert torch.all(out_img[:, 8:, :] == 0)
    assert torch.all(out_img[:, :, 8:] == 0)
    
    # Check values in valid region
    assert torch.all(out_img[:, :8, :8] == 0.5)

def test_patch_aligned_repeat_invalid():
    image = torch.zeros((3, 64, 64))
    valid_node_mask = torch.zeros(64, dtype=torch.bool)
    valid_node_mask[:44] = True
    
    import pytest
    with pytest.raises(ValueError) as exc:
        transform_graph_image_for_model(
            image=image,
            valid_node_mask=valid_node_mask,
            input_mode="patch_aligned_repeat",
            patch_size=16,
            token_grid_size=32
        )
    assert "n_valid=44" in str(exc.value)
    assert "token_grid_size=32" in str(exc.value)
