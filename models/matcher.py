import torch
import torch.nn as nn
import pygmtools


class GraphMatcher(nn.Module):
    """
    Graph matching model with Sinkhorn.
    Matches MatchingModel_MLPGATv2Sinkhorn architecture
    """
    def __init__(self, encoder, params=None):
        super().__init__()
        self.encoder = encoder

        if params is None:
            from config import ModelParams
            params = ModelParams.get_default()

        self.params = params
        self.sinkhorn_max_iter = params.sinkhorn_iterations
        self.sinkhorn_tau = params.sinkhorn_tau
        self.inst_norm = nn.InstanceNorm2d(1, affine=True)

    def forward(self, batch1, batch2, perm_list=None, batch_idx1=None, batch_idx2=None, inference=False):
        """
        Forward pass for batched graph matching.
        """
        device = next(self.parameters()).device

        x1, edge1 = batch1.x.to(device), batch1.edge_index.to(device)
        x2, edge2 = batch2.x.to(device), batch2.edge_index.to(device)

        batch_idx1 = batch1.batch.to(device) if batch_idx1 is None else batch_idx1.to(device)
        batch_idx2 = batch2.batch.to(device) if batch_idx2 is None else batch_idx2.to(device)

        h1 = self.encoder(batch1)
        h2 = self.encoder(batch2)

        B = batch_idx1.max().item() + 1
        perm_pred_list = []
        all_embeddings = []

        for b in range(B):
            h1_b = h1[batch_idx1 == b]
            h2_b = h2[batch_idx2 == b]
            N1, N2 = h1_b.size(0), h2_b.size(0)

            sim = torch.matmul(h1_b, h2_b.T)
            sim_batched = sim.unsqueeze(0).unsqueeze(1)
            sim_normed = self.inst_norm(sim_batched).squeeze(1)

            transposed = N1 > N2

            if transposed:
                sim_input = sim_normed.transpose(-2, -1)
                nr = torch.tensor([N2], dtype=torch.long, device=device)
                nc = torch.tensor([N1], dtype=torch.long, device=device)
            else:
                sim_input = sim_normed
                nr = torch.tensor([N1], dtype=torch.long, device=device)
                nc = torch.tensor([N2], dtype=torch.long, device=device)

            S = pygmtools.sinkhorn(
                sim_input,
                n1=nr, n2=nc,
                dummy_row=(N1 != N2),
                max_iter=self.sinkhorn_max_iter,
                tau=self.sinkhorn_tau
            )

            if transposed:
                S = S.transpose(-2, -1)

            perm_pred_list.append(S.squeeze(0))
            all_embeddings.append((h1_b, h2_b))

        return perm_pred_list, all_embeddings

    def get_hard_assignment(self, S, N2):
        """Extract hard assignment using Hungarian algorithm."""
        return pygmtools.hungarian(S)


class MatchingModel_MLPGATv2Sinkhorn(nn.Module):
    """
    MLP + GATv2 + Sinkhorn model for graph matching.
    """
    def __init__(self, params=None):
        super().__init__()

        if params is None:
            from config import ModelParams
            params = ModelParams.get_default()

        self.params = params

        in_dim = params.input_dim
        hidden_dim = params.hidden_dim
        out_dim = params.output_dim
        sinkhorn_max_iter = params.sinkhorn_iterations
        sinkhorn_tau = params.sinkhorn_tau
        attention_dropout = params.attn_dropout
        dropout_emb = params.dropout
        num_layers = params.num_layers
        heads = params.num_heads

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_emb),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_emb)
        )

        self.gnn = nn.ModuleList()
        dims = [hidden_dim] * num_layers + [out_dim]
        for i in range(num_layers):
            self.gnn.append(
                GATv2Conv(dims[i], dims[i+1],
                          heads=heads, concat=False,
                          dropout=attention_dropout)
            )

        self.dropout = nn.Dropout(p=dropout_emb)
        self.inst_norm = nn.InstanceNorm2d(1, affine=True)
        self.sinkhorn_max_iter = sinkhorn_max_iter
        self.sinkhorn_tau = sinkhorn_tau

    def encode(self, x, edge_index):
        for i, conv in enumerate(self.gnn):
            x = conv(x, edge_index)
            if i < len(self.gnn) - 1:
                x = F.relu(x)
                x = self.dropout(x)
        return x

    def forward(self, batch1, batch2, perm_list=None, batch_idx1=None, batch_idx2=None, inference=False):
        device = next(self.parameters()).device
        x1, edge1 = batch1.x.to(device), batch1.edge_index.to(device)
        x2, edge2 = batch2.x.to(device), batch2.edge_index.to(device)
        perm_list = [p.to(device) for p in perm_list] if perm_list is not None else None

        batch_idx1 = batch1.batch.to(device) if batch_idx1 is None else batch_idx1.to(device)
        batch_idx2 = batch2.batch.to(device) if batch_idx2 is None else batch_idx2.to(device)

        h1 = self.mlp(x1)
        h2 = self.mlp(x2)
        h1 = self.encode(h1, edge1)
        h2 = self.encode(h2, edge2)

        B = batch_idx1.max().item() + 1
        perm_pred_list = []
        all_embeddings = []

        for b in range(B):
            h1_b = h1[batch_idx1 == b]
            h2_b = h2[batch_idx2 == b]
            N1, N2 = h1_b.size(0), h2_b.size(0)

            sim = torch.matmul(h1_b, h2_b.T)
            sim_batched = sim.unsqueeze(0).unsqueeze(1)
            sim_normed = self.inst_norm(sim_batched).squeeze(1)

            transposed = N1 > N2

            if transposed:
                sim_input = sim_normed.transpose(-2, -1)
                nr = torch.tensor([N2], dtype=torch.long, device=device)
                nc = torch.tensor([N1], dtype=torch.long, device=device)
            else:
                sim_input = sim_normed
                nr = torch.tensor([N1], dtype=torch.long, device=device)
                nc = torch.tensor([N2], dtype=torch.long, device=device)

            S = pygmtools.sinkhorn(
                sim_input,
                n1=nr, n2=nc,
                dummy_row=(N1 != N2),
                max_iter=self.sinkhorn_max_iter,
                tau=self.sinkhorn_tau
            )

            if transposed:
                S = S.transpose(-2, -1)

            perm_pred_list.append(S.squeeze(0))
            all_embeddings.append((h1_b, h2_b))

        return perm_pred_list, all_embeddings