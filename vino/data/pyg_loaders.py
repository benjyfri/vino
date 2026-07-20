import torch
import numpy as np
from sklearn.model_selection import train_test_split
from .graph_record import GraphRecord
from typing import List, Sequence


def scaffold_split_indices(smiles: Sequence[str], seed: int = 42) -> dict[str, list[int]]:
    """Deterministic Bemis-Murcko scaffold split with disjoint scaffold groups."""
    try:
        from rdkit.Chem.Scaffolds import MurckoScaffold
    except ImportError as exc:
        raise ImportError("rdkit is required for scaffold splitting") from exc
    groups: dict[str, list[int]] = {}
    for index, value in enumerate(smiles):
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(smiles=str(value), includeChirality=True)
        # Acyclic molecules have an empty Murcko scaffold. Keep their exact
        # structures separate rather than collapsing all of them into one group.
        key = scaffold or f"__acyclic__:{value}"
        groups.setdefault(key, []).append(index)
    rng = np.random.default_rng(seed)
    ordered = list(groups.items())
    rng.shuffle(ordered)
    ordered.sort(key=lambda item: len(item[1]), reverse=True)
    targets = {"train": 0.8 * len(smiles), "val": 0.1 * len(smiles), "test": 0.1 * len(smiles)}
    result = {name: [] for name in targets}
    for _, indices in ordered:
        split = max(targets, key=lambda name: targets[name] - len(result[name]))
        result[split].extend(indices)
    return {name: sorted(indices) for name, indices in result.items()}

def load_bbbp(limit=None, split_seed: int = 42, split_strategy: str = "scaffold",
              root: str = "data/pyg") -> List[GraphRecord]:
    try:
        from torch_geometric.datasets import MoleculeNet
    except ImportError:
        raise ImportError("torch_geometric is required to load BBBP. Please install it via 'pip install torch_geometric'.")
        
    dataset = MoleculeNet(root=root, name="BBBP")
    
    records = []
    
    indices = np.arange(len(dataset))
    if split_strategy == "scaffold":
        smiles = getattr(dataset, "smiles", None)
        if smiles is None or len(smiles) != len(dataset):
            raise ValueError("BBBP scaffold split requires dataset.smiles for every graph")
        split_indices = scaffold_split_indices(smiles, seed=split_seed)
        train_idx, val_idx, test_idx = (split_indices[name] for name in ("train", "val", "test"))
    elif split_strategy == "stratified_random":
        labels = np.asarray([int(dataset[i].y.reshape(-1)[0]) for i in indices])
        train_idx, held_idx = train_test_split(
            indices, test_size=0.2, random_state=split_seed, stratify=labels
        )
        val_idx, test_idx = train_test_split(
            held_idx, test_size=0.5, random_state=split_seed, stratify=labels[held_idx]
        )
    else:
        raise ValueError(f"Unsupported BBBP split strategy: {split_strategy!r}")
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
            metadata={"source_index": i, "split_strategy": split_strategy, "split_seed": split_seed},
        ))
        
    return records


def load_esol(root: str = "data/pyg", split_seed: int = 42) -> List[GraphRecord]:
    try:
        from torch_geometric.datasets import MoleculeNet
    except ImportError as exc:
        raise ImportError("torch_geometric is required to load ESOL") from exc
    dataset = MoleculeNet(root=root, name="ESOL")
    smiles = getattr(dataset, "smiles", None)
    if smiles is None:
        raise ValueError("ESOL scaffold split requires dataset.smiles")
    split_indices = scaffold_split_indices(smiles, split_seed)
    split_by_index = {i: split for split, values in split_indices.items() for i in values}
    return [GraphRecord(
        graph_id=f"esol_{i}", x=data.x.float(), edge_index=data.edge_index,
        edge_attr=data.edge_attr.float() if data.edge_attr is not None else None,
        y=data.y.float(), split=split_by_index[i], num_nodes=data.num_nodes,
        metadata={"source_index": i, "split_strategy": "scaffold", "split_seed": split_seed},
    ) for i, data in enumerate(dataset)]

def load_molhiv(root: str = "data/ogb") -> List[GraphRecord]:
    try:
        from ogb.graphproppred import PygGraphPropPredDataset
    except ImportError as exc:
        raise ImportError("ogb is required to load ogbg-molhiv") from exc
    dataset = PygGraphPropPredDataset(name="ogbg-molhiv", root=root)
    split_idx = dataset.get_idx_split()
    # OGB names the validation split "valid"; VINO's internal contract (GraphRecord,
    # build_graph_images) uses "val". Translate before assigning record splits.
    ogb_split_to_internal = {"train": "train", "valid": "val", "test": "test"}
    split_by_index = {
        int(i): ogb_split_to_internal[split]
        for split, values in split_idx.items()
        for i in values.reshape(-1).tolist()
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
