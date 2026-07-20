import torch.nn as nn


def set_frozen_modules_eval(module: nn.Module) -> None:
    """Put every submodule whose parameters are all frozen into ``eval()`` mode.

    For partially-unfrozen backbones (``last1``/``last2``/``train_patch_embed``) the parent
    model is in ``train()`` mode so the trainable layers behave correctly, but that also
    leaves stochastic ops (Dropout, DropPath) and normalization running-stat updates active
    inside the *frozen* layers, perturbing features that are supposed to be fixed. Recurse
    into mixed modules and switch fully-frozen submodules (and their children) to eval.
    """
    for child in module.children():
        params = list(child.parameters(recurse=True))
        if params and all(not p.requires_grad for p in params):
            child.eval()
        else:
            set_frozen_modules_eval(child)
