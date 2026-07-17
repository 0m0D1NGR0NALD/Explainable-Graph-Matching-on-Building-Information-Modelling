import os
import pickle
from typing import List, Tuple

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch


def nx_to_pyg_data_preserve_order(graph) -> Data:
    """
    Convert a NetworkX DiGraph to a PyTorch Geometric Data object,
    preserving node insertion order.
    """
    node_ids = list(graph.nodes())
    id_map = {nid: i for i, nid in enumerate(node_ids)}

    node_type_mapping = {"room": [1, 0], "ws": [0, 1]}

    x = torch.stack([
        torch.tensor(
            node_type_mapping[graph.nodes[n]['type']] +
            graph.nodes[n]['center'] +
            graph.nodes[n]['normal'] +
            [graph.nodes[n].get('length', -1)],
            dtype=torch.float32
        )
        for n in node_ids
    ])

    edge_index = torch.tensor(
        [[id_map[u], id_map[v]] for u, v in graph.edges()],
        dtype=torch.long
    ).t().contiguous() if graph.edges else torch.empty((2, 0), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index)
    data.name = graph.graph.get('name', '')
    data.node_names = node_ids
    data.permutation = torch.arange(len(node_ids), dtype=torch.long)
    return data


def pyg_data_to_nx_digraph(data: Data, graph_list: List) -> object:
    """
    Convert a PyTorch Geometric Data object back to a NetworkX DiGraph.
    """
    import networkx as nx

    assert hasattr(data, 'node_names'), \
        "Data object must contain 'node_names' to restore original node IDs."
    assert hasattr(data, 'permutation'), \
        "Data object must contain 'permutation' to reorder nodes."
    assert hasattr(data, 'name'), \
        "Data object must contain 'name' to match with graph_list."

    candidates = [g for g in graph_list if g.graph.get('name') == data.name]

    if not candidates:
        raise ValueError(f"No graph with name {data.name} found in graph_list.")

    node_name_set = set(data.node_names)
    matching_graph = next((g for g in candidates if set(g.nodes()) == node_name_set), None)

    if matching_graph is None:
        matching_graph = max(candidates, key=lambda g: len(node_name_set & set(g.nodes())))

    orig_names = data.node_names
    perm = data.permutation.tolist()
    node_ids = [orig_names[idx] for idx in perm]

    G = nx.DiGraph()
    for node_id in node_ids:
        if node_id in matching_graph.nodes:
            G.add_node(node_id, **matching_graph.nodes[node_id])

    for u_idx, v_idx in data.edge_index.t().tolist():
        u = node_ids[u_idx]
        v = node_ids[v_idx]
        if matching_graph.has_edge(u, v):
            G.add_edge(u, v, **matching_graph.edges[u, v])

    G.graph['name'] = data.name
    return G


def deserialize_graph_matching_dataset(path: str, filename: str = "train_dataset.pkl") -> List[Tuple[Data, Data, torch.Tensor]]:
    """Deserialize dataset of (Data1, Data2, PermutationMatrix) tuples"""
    full_path = os.path.join(path, filename)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"File not found: {full_path}")
    with open(full_path, 'rb') as f:
        pairs = pickle.load(f)
    print(f"Loaded {len(pairs)} pairs from {full_path}")
    return pairs


def serialize_graph_matching_dataset(pairs: List[Tuple[Data, Data, torch.Tensor]], path: str, filename: str = "dataset.pkl"):
    """Serialize dataset to file"""
    os.makedirs(path, exist_ok=True)
    full_path = os.path.join(path, filename)
    with open(full_path, 'wb') as f:
        pickle.dump(pairs, f)
    print(f"Serialized {len(pairs)} pairs to {full_path}")


def compute_mean_std(pairs: List[Tuple[Data, Data, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute per-feature mean and std from training set"""
    x_list = []
    for data1, data2, _ in pairs:
        x_list.append(data1.x)
        x_list.append(data2.x)
    x_all = torch.cat(x_list, dim=0)
    mean = x_all.mean(dim=0)
    std = x_all.std(dim=0)
    return mean, std


def normalize_data_pairs(pairs: List[Tuple[Data, Data, torch.Tensor]], mean: torch.Tensor, std: torch.Tensor) -> List[Tuple[Data, Data, torch.Tensor]]:
    """Normalize features in Data objects."""
    normalized_pairs = []
    for data1, data2, P in pairs:
        data1_clone = Data(x=(data1.x - mean) / (std + 1e-8), edge_index=data1.edge_index)
        data2_clone = Data(x=(data2.x - mean) / (std + 1e-8), edge_index=data2.edge_index)
        if hasattr(data1, 'name'):
            data1_clone.name = data1.name
        if hasattr(data2, 'name'):
            data2_clone.name = data2.name
        normalized_pairs.append((data1_clone, data2_clone, P))
    return normalized_pairs


def collate_pyg_matching(batch):
    """Custom collate function for graph matching batches"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data1_list, data2_list, perm_list = zip(*batch)

    data1_list = [d.to(device) for d in data1_list]
    data2_list = [d.to(device) for d in data2_list]

    batch1 = Batch.from_data_list(data1_list)
    batch2 = Batch.from_data_list(data2_list)

    return batch1, batch2, perm_list


class GraphMatchingDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]