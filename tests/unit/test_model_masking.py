import torch
from omegaconf import OmegaConf
from vino.models.graph_image_model import GraphImageModel
from vino.utils.config import load_resolved_config


def test_invalid_patch_values_do_not_affect_prediction():
    cfg = load_resolved_config("configs/experiments/smoke_synthetic.yaml")
    resolved = OmegaConf.to_container(cfg, resolve=True)
    torch.manual_seed(3)
    model = GraphImageModel(resolved).eval()
    first = torch.zeros(1, 3, 32, 32)
    second = first.clone()
    second[:, :, 16:, :] = 1.0
    mask = torch.tensor([[[True, True], [False, False]]])
    with torch.no_grad():
        assert torch.allclose(model(first, mask), model(second, mask))
