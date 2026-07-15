import torch
from .graph_record import GraphRecord
from typing import List

def generate_synthetic_graphs(
    num_graphs: int,
    min_nodes: int,
    max_nodes: int,
    node_dim: int,
    edge_dim: int,
    task_type: str = "binary_classification",
    seed: int = 42
) -> List[GraphRecord]:
    torch.manual_seed(seed)
    records = []
    
    for i in range(num_graphs):
        n = torch.randint(min_nodes, max_nodes + 1, (1,)).item()
        
        # Random node features
        x = torch.randn(n, node_dim)
        
        # Erdos-Renyi like edges, but ensure at least some edges
        p = 0.3
        adj = (torch.rand(n, n) < p).int()
        adj.fill_diagonal_(0)
        adj = torch.max(adj, adj.T) # symmetric
        
        edge_index = adj.nonzero(as_tuple=False).t().contiguous()
        
        m = edge_index.size(1)
        if m > 0:
            edge_attr = torch.randn(m, edge_dim)
        else:
            edge_attr = torch.zeros(0, edge_dim)
            
        # Determine label based on topology (e.g., number of edges) and features
        signal = x.mean() + m / (n * n)
        
        if task_type == "binary_classification":
            y = (signal > 0).long().view(1)
        else:
            y = signal.view(1)
            
        split = "train"
        if i >= int(num_graphs * 0.8):
            split = "test"
        elif i >= int(num_graphs * 0.6):
            split = "val"
            
        records.append(GraphRecord(
            graph_id=f"synth_{i}",
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            split=split,
            num_nodes=n
        ))
        
    return records
