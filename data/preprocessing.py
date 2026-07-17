import torch
from torch_geometric.data import Data


node_type_mapping = {"room": [1, 0], "ws": [0, 1]}


def add_edge_types_to_graph(data):
    """Add edge type information to PyG Data object."""
    if not hasattr(data, 'node_names') or data.node_names is None:
        return data

    node_names = data.node_names

    edge_types = []
    for u_idx, v_idx in data.edge_index.t().tolist():
        if u_idx >= len(node_names) or v_idx >= len(node_names):
            edge_types.append(0)
            continue

        u_name = node_names[u_idx]
        v_name = node_names[v_idx]

        u_is_room = 'centroid' in u_name or ('room' in u_name.lower() and 'ws' not in u_name)
        v_is_room = 'centroid' in v_name or ('room' in v_name.lower() and 'ws' not in v_name)

        if u_is_room != v_is_room:
            edge_types.append(0)
        elif u_is_room and v_is_room:
            edge_types.append(1)
        else:
            edge_types.append(2)

    data.edge_type = torch.tensor(edge_types, dtype=torch.long)
    return data


def normalize_on_fly(g, mean, std):
    """Apply normalization while preserving attributes."""
    g_norm = Data(x=(g.x - mean) / (std + 1e-8), edge_index=g.edge_index)
    if hasattr(g, 'name'):
        g_norm.name = g.name
    if hasattr(g, 'node_names'):
        g_norm.node_names = g.node_names
    if hasattr(g, 'permutation'):
        g_norm.permutation = g.permutation
    if hasattr(g, 'edge_type'):
        g_norm.edge_type = g.edge_type
    return g_norm