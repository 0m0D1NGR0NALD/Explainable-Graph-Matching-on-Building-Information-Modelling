import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from config import ModelParams


class FeatureHomogenizer(nn.Module):
    """
    MLP to homogenize heterogeneous node features.
    As described in paper Section III-A: two-layer MLP with ReLU and dropout.
    """
    def __init__(self, params: ModelParams = None):
        super().__init__()
        if params is None:
            params = ModelParams.get_default()

        self.mlp = nn.Sequential(
            nn.Linear(params.input_dim, params.hidden_dim),
            nn.ReLU(),
            nn.Dropout(params.dropout),
            nn.Linear(params.hidden_dim, params.hidden_dim),
            nn.ReLU(),
            nn.Dropout(params.dropout)
        )

    def forward(self, x):
        return self.mlp(x)


class BaseGNNEncoder(nn.Module, ABC):
    """Base class for all GNN encoders."""
    def __init__(self, params: ModelParams = None):
        super().__init__()
        if params is None:
            params = ModelParams.get_default()

        self.params = params
        self.homogenizer = FeatureHomogenizer(params)
        self.output_dim = params.output_dim
        self.dropout = params.dropout

    @abstractmethod
    def forward(self, data):
        pass