import torch
import numpy as np
import random
from vino.utils.seed import seed_everything

def test_seed_everything():
    seed_everything(42)
    val1 = random.randint(0, 1000)
    val2 = np.random.rand()
    val3 = torch.randn(1).item()
    
    seed_everything(42)
    assert val1 == random.randint(0, 1000)
    assert val2 == np.random.rand()
    assert val3 == torch.randn(1).item()
