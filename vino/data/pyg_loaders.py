import torch
import numpy as np
from sklearn.model_selection import train_test_split
from .graph_record import GraphRecord
from typing import List

def load_bbbp(limit=None, split_seed: int = 42) -> List[GraphRecord]:
    try:
        from torch_geometric.datasets import MoleculeNet
    except ImportError:
        raise ImportError("torch_geometric is required to load BBBP. Please install it via 'pip install torch_geometric'.")
        
    dataset = MoleculeNet(root="data/pyg", name="BBBP")
    
    records = []
    
    indices = np.arange(len(dataset))
    labels = np.asarray([int(dataset[i].y.reshape(-1)[0]) for i in indices])
    train_idx, held_idx = train_test_split(
        indices, test_size=0.2, random_state=split_seed, stratify=labels
    )
    val_idx, test_idx = train_test_split(
        held_idx, test_size=0.5, random_state=split_seed, stratify=labels[held_idx]
    )
    split_by_index = {int(i): "train" for i in train_idx}
    split_by_index.update({int(i): "val" for i in val_idx})
    split_by_index.update({int(i): "test" for i in test_idx})
    selected = indices if limit is None else indices[:min(int(limit), len(indices))]
        
    for i in selected:
        i = int(i)
        data = dataset[i]
        
        split = split_by_index[i]
            
        x = data.x.float() if hasattr(data, 'x') and data.x is not None else torch.zeros(data.num_nodes, 1)
        edge_index = data.edge_index
        if hasattr(data, "edge_attr") and data.edge_attr is not None:
            edge_attr = data.edge_attr.float()
        else:
            m = edge_index.size(1)
            edge_attr = torch.ones(m, 1)
            
        y = data.y.float()
        
        records.append(GraphRecord(
            graph_id=f"bbbp_{i}",
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            split=split,
            num_nodes=data.num_nodes,
            metadata={"source_index": i, "split_strategy": "stratified_random", "split_seed": split_seed},
        ))
        
    return records

def load_molhiv(root: str = "data/ogb") -> List[GraphRecord]:
    try:
        from ogb.graphproppred import PygGraphPropPredDataset
    except ImportError as exc:
        raise ImportError("ogb is required to load ogbg-molhiv") from exc
    dataset = PygGraphPropPredDataset(name="ogbg-molhiv", root=root)
    split_idx = dataset.get_idx_split()
    split_by_index = {
        int(i): split for split, values in split_idx.items() for i in values.reshape(-1).tolist()
    }
    records = []
    for i in range(len(dataset)):
        data = dataset[i]
        records.append(GraphRecord(
            graph_id=f"molhiv_{i}", x=data.x.float(), edge_index=data.edge_index,
            edge_attr=data.edge_attr.float() if data.edge_attr is not None else None,
            y=data.y.float(), split=split_by_index[i], num_nodes=data.num_nodes,
            metadata={"source_index": i, "split_strategy": "ogb_official"},
        ))
    return records
