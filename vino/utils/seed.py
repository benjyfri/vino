import torch
import numpy as np
import random
import os

def seed_everything(seed: int = 42) -> None:
    """Sets the seed for deterministic behavior."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Make cudnn deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Optional: use deterministic algorithms
    # torch.use_deterministic_algorithms(True, warn_only=True)
