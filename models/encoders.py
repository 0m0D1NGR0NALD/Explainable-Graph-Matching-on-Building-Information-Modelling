import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv, SAGEConv, GINConv, GATv2Conv, 
    TransformerConv, GINEConv, RGCNConv
)

from .base import BaseGNNEncoder
from config import ModelParams


class GCNEncoder(BaseGNNEncoder):
    """GCN encoder."""
    def __init__(self, params: ModelParams = None):
        super().__init__(params)
        if params is None:
            params = ModelParams.get_default()

        self.conv1 = GCNConv(params.hidden_dim, params.hidden_dim)
        self.conv2 = GCNConv(params.hidden_dim, params.output_dim)
        self.dropout = nn.Dropout(p=params.dropout)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.homogenizer(x)

        x = self.dropout(x)

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)

        return x


class GraphSAGEEncoder(BaseGNNEncoder):
    """GraphSAGE encoder."""
    def __init__(self, params: ModelParams = None):
        super().__init__(params)
        if params is None:
            params = ModelParams.get_default()

        self.conv1 = SAGEConv(params.hidden_dim, params.hidden_dim)
        self.conv2 = SAGEConv(params.hidden_dim, params.output_dim)
        self.dropout = nn.Dropout(p=params.dropout)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.homogenizer(x)

        x = self.dropout(x)

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)

        return x


class GINEncoder(BaseGNNEncoder):
    """GIN encoder."""
    def __init__(self, params: ModelParams = None):
        super().__init__(params)
        if params is None:
            params = ModelParams.get_default()

        mlp1 = nn.Sequential(
            nn.Linear(params.hidden_dim, params.hidden_dim),
            nn.ReLU(),
            nn.Linear(params.hidden_dim, params.hidden_dim)
        )
        mlp2 = nn.Sequential(
            nn.Linear(params.hidden_dim, params.hidden_dim),
            nn.ReLU(),
            nn.Linear(params.hidden_dim, params.output_dim)
        )

        self.conv1 = GINConv(mlp1)
        self.conv2 = GINConv(mlp2)
        self.dropout = nn.Dropout(p=params.dropout)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.homogenizer(x)

        x = self.dropout(x)

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)

        return x


class GraphTransformerEncoder(BaseGNNEncoder):
    """Graph Transformer encoder."""
    def __init__(self, params: ModelParams = None):
        super().__init__(params)
        if params is None:
            params = ModelParams.get_default()

        self.conv1 = TransformerConv(
            in_channels=params.hidden_dim,
            out_channels=params.hidden_dim,
            heads=params.num_heads,
            dropout=params.attn_dropout,
            concat=False,
            edge_dim=params.hidden_dim // 4,
            beta=True
        )

        self.conv2 = TransformerConv(
            in_channels=params.hidden_dim,
            out_channels=params.output_dim,
            heads=params.num_heads,
            dropout=params.attn_dropout,
            concat=False,
            edge_dim=params.hidden_dim // 4,
            beta=True
        )

        self.edge_proj = nn.Linear(1, params.hidden_dim // 4)
        self.dropout = nn.Dropout(p=params.dropout)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.homogenizer(x)
        x = self.dropout(x)

        edge_attr = None
        if hasattr(data, 'edge_type'):
            edge_type_float = data.edge_type.float().unsqueeze(-1)
            edge_attr = self.edge_proj(edge_type_float)

        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index, edge_attr=edge_attr)

        return x


class GATv2Encoder(BaseGNNEncoder):
    """GATv2 encoder."""
    def __init__(self, params: ModelParams = None):
        super().__init__(params)
        if params is None:
            params = ModelParams.get_default()

        self.conv1 = GATv2Conv(
            params.hidden_dim, params.hidden_dim,
            heads=4,
            dropout=params.attn_dropout,
            concat=False
        )

        self.conv2 = GATv2Conv(
            params.hidden_dim, params.output_dim,
            heads=4,
            dropout=params.attn_dropout,
            concat=False
        )

        self.dropout = nn.Dropout(p=params.dropout)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.homogenizer(x)

        x = self.dropout(x)

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)

        return x


class GINEEncoder(BaseGNNEncoder):
    """GINE (Graph Isomorphism Network with Edge Features) encoder."""
    def __init__(self, params: ModelParams = None):
        super().__init__(params)
        if params is None:
            params = ModelParams.get_default()

        self.edge_embedding = nn.Embedding(3, params.hidden_dim // 4)
        edge_dim = params.hidden_dim // 4

        mlp1 = nn.Sequential(
            nn.Linear(params.hidden_dim, params.hidden_dim),
            nn.ReLU(),
            nn.Linear(params.hidden_dim, params.hidden_dim)
        )
        self.conv1 = GINEConv(mlp1, edge_dim=edge_dim)

        mlp2 = nn.Sequential(
            nn.Linear(params.hidden_dim, params.hidden_dim),
            nn.ReLU(),
            nn.Linear(params.hidden_dim, params.output_dim)
        )
        self.conv2 = GINEConv(mlp2, edge_dim=edge_dim)

        self.dropout = nn.Dropout(p=params.dropout)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.homogenizer(x)

        x = self.dropout(x)

        edge_attr = None
        if hasattr(data, 'edge_type'):
            edge_attr = self.edge_embedding(data.edge_type)

        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index, edge_attr=edge_attr)

        return x


class RGCNEncoder(BaseGNNEncoder):
    """RGCN (Relational Graph Convolutional Network) encoder."""
    def __init__(self, params: ModelParams = None):
        super().__init__(params)
        if params is None:
            params = ModelParams.get_default()

        self.conv1 = RGCNConv(
            in_channels=params.hidden_dim,
            out_channels=params.hidden_dim,
            num_relations=3
        )

        self.conv2 = RGCNConv(
            in_channels=params.hidden_dim,
            out_channels=params.output_dim,
            num_relations=3
        )

        self.dropout = nn.Dropout(p=params.dropout)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.homogenizer(x)

        x = self.dropout(x)

        edge_type = data.edge_type if hasattr(data, 'edge_type') else None

        x = self.conv1(x, edge_index, edge_type)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index, edge_type)

        return x