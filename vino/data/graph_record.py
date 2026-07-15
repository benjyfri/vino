from dataclasses import dataclass, field
import torch
from typing import Optional, Dict, Any

@dataclass
class GraphRecord:
    graph_id: str
    x: torch.Tensor
    edge_index: torch.LongTensor
    y: torch.Tensor
    edge_attr: Optional[torch.Tensor] = None
    split: Optional[str] = None
    num_nodes: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.num_nodes == 0:
            if self.x is not None and self.x.dim() > 0:
                self.num_nodes = self.x.size(0)
            elif self.edge_index is not None and self.edge_index.numel() > 0:
                self.num_nodes = self.edge_index.max().item() + 1
