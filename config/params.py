from dataclasses import dataclass, asdict
from typing import Dict, Any
import json


@dataclass
class ModelParams:
    """Unified parameters for all GNN models."""

    # Data Parameters
    input_dim: int = 7
    batch_size: int = 8

    # GNN Architecture Parameters
    hidden_dim: int = 128
    output_dim: int = 16
    num_layers: int = 2
    num_heads: int = 4

    # Regularization Parameters
    dropout: float = 0.0001792177005695561
    attn_dropout: float = 0.00609918816039232

    # Sinkhorn Parameters
    sinkhorn_iterations: int = 73
    sinkhorn_tau: float = 0.999101821286028

    # Optimization Parameters
    learning_rate: float = 0.0023737917298792635
    weight_decay: float = 3.272922404797929e-05

    # Training Parameters
    num_epochs: int = 100
    patience: int = 10

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def to_json(self, filepath: str):
        """Save parameters to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, filepath: str) -> 'ModelParams':
        """Load parameters from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls(**data)

    @classmethod
    def get_default(cls) -> 'ModelParams':
        """Get default parameters (optimal from paper author)."""
        return cls()

    @classmethod
    def get_paper_params(cls) -> 'ModelParams':
        """Get published paper parameters (non-optimal)."""
        return cls(
            hidden_dim=64,
            output_dim=32,
            dropout=0.15,
            attn_dropout=0.12,
            sinkhorn_iterations=20,
            sinkhorn_tau=1.0,
            learning_rate=0.001,
            weight_decay=5e-5,
            batch_size=16
        )