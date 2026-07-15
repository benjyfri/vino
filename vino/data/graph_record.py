from dataclasses import dataclass, field
import torch
from typing import Optional, Dict, Any

@dataclass
class GraphRecord:
    graph_id: str
    x: Optional[torch.Tensor]
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
        self.validate()

    def validate(self) -> None:
        if not isinstance(self.graph_id, str) or not self.graph_id:
            raise ValueError("graph_id must be a non-empty string")
        if not isinstance(self.num_nodes, int) or self.num_nodes < 0:
            raise ValueError(f"num_nodes must be a non-negative integer, got {self.num_nodes!r}")
        if self.edge_index is None or self.edge_index.dtype != torch.long:
            raise ValueError("edge_index must be a torch.long tensor")
        if self.edge_index.ndim != 2 or self.edge_index.shape[0] != 2:
            raise ValueError(f"edge_index must have shape [2, M], got {tuple(self.edge_index.shape)}")
        if self.edge_index.numel():
            minimum, maximum = int(self.edge_index.min()), int(self.edge_index.max())
            if minimum < 0 or maximum >= self.num_nodes:
                raise ValueError(f"edge_index values must be in [0, {self.num_nodes}), got [{minimum}, {maximum}]")
        if self.x is not None:
            if self.x.ndim != 2 or self.x.shape[0] != self.num_nodes:
                raise ValueError(f"x must have shape [{self.num_nodes}, D], got {tuple(self.x.shape)}")
            if not torch.isfinite(self.x).all():
                raise ValueError("x contains non-finite values")
        if self.edge_attr is not None:
            if self.edge_attr.ndim != 2 or self.edge_attr.shape[0] != self.edge_index.shape[1]:
                raise ValueError("edge_attr rows must match edge_index columns")
            if not torch.isfinite(self.edge_attr).all():
                raise ValueError("edge_attr contains non-finite values")
        if self.y is None or not isinstance(self.y, torch.Tensor) or self.y.numel() == 0:
            raise ValueError("y must be a non-empty tensor")
        if self.split not in {None, "train", "val", "test"}:
            raise ValueError(f"split must be train/val/test/None, got {self.split!r}")
