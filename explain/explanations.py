import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.explain import Explainer, GNNExplainer
import pygmtools

from data import add_edge_types_to_graph


class NodeClassifier(nn.Module):
    """Node classifier wrapper for GNN encoder."""
    def __init__(self, encoder, num_classes, edge_type):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(encoder.output_dim, num_classes)
        self.edge_type = edge_type

    def forward(self, x, edge_index, edge_attr=None):
        data = Data(x=x, edge_index=edge_index)

        if self.edge_type is not None:
            data.edge_type = self.edge_type
        elif edge_attr is not None:
            data.edge_type = edge_attr

        h = self.encoder(data)
        return self.classifier(h)


def get_misclassified_node_and_predictions(model, g1_orig, g2_orig, P_gt, device, mean, std):
    """
    Get hard assignments from GNN matching and find misclassified matches.
    """
    print(f"\n  [DEBUG] get_misclassified_node_and_predictions:")
    print(f"    S-graph (g2_orig) has {g2_orig.x.shape[0]} nodes (indices 0 to {g2_orig.x.shape[0]-1})")
    print(f"    A-graph (g1_orig) has {g1_orig.x.shape[0]} nodes (indices 0 to {g1_orig.x.shape[0]-1})")
    print(f"    P_gt shape: {P_gt.shape} (rows=A-nodes, cols=S-nodes)")

    mean = mean.to(device)
    std = std.to(device)

    def normalize(g):
        g_norm = Data(
            x=(g.x.to(device) - mean) / (std + 1e-8),
            edge_index=g.edge_index.to(device)
        )
        if hasattr(g, 'node_names'):
            g_norm.node_names = g.node_names
        if hasattr(g, 'name'):
            g_norm.name = g.name
        if hasattr(g, 'permutation'):
            g_norm.permutation = g.permutation
        if hasattr(g, 'edge_type'):
            g_norm.edge_type = g.edge_type.to(device) if g.edge_type is not None else None
        if hasattr(g, 'node_type'):
            g_norm.node_type = g.node_type.to(device) if g.node_type is not None else None
        return g_norm

    g1_norm = normalize(g1_orig)
    g2_norm = normalize(g2_orig)

    model.eval()
    with torch.no_grad():
        batch1 = Batch.from_data_list([g1_norm])
        batch2 = Batch.from_data_list([g2_norm])
        S_pred_list, _ = model(batch1, batch2)
        S_pred = S_pred_list[0]

        N_A, N_S = P_gt.shape
        S_real = S_pred[:, :N_S]

        hard_assign_a_to_s = pygmtools.hungarian(S_real.unsqueeze(0)).squeeze(0)
        hard_assign_s_to_a = hard_assign_a_to_s.T

        pred_labels = hard_assign_s_to_a.argmax(dim=1)
        true_labels = P_gt.T.argmax(dim=1)

    misclassified_s_nodes = []
    for s_idx in range(N_S):
        if pred_labels[s_idx].item() != true_labels[s_idx].item():
            misclassified_s_nodes.append(s_idx)

    print(f"    misclassified S-nodes found: {misclassified_s_nodes}")
    if misclassified_s_nodes:
        print(f"    First misclassified S-node: {misclassified_s_nodes[0]} -> predicts A-{pred_labels[misclassified_s_nodes[0]].item()}, should be A-{true_labels[misclassified_s_nodes[0]].item()}")

    return hard_assign_s_to_a.cpu(), pred_labels.cpu(), true_labels.cpu(), misclassified_s_nodes


def get_gnnexplainer_importance_for_misclassified(model, g1_orig, g2_orig, P_gt, device, mean, std, epochs=100):
    """
    Use GNNExplainer to explain a WRONGLY matched S-node.
    """
    print(f"\n  [DEBUG] get_gnnexplainer_importance_for_misclassified:")
    print(f"    S-graph has {g2_orig.x.shape[0]} nodes")
    print(f"    A-graph has {g1_orig.x.shape[0]} nodes")

    if not hasattr(g1_orig, 'edge_type'):
        g1_orig = add_edge_types_to_graph(g1_orig)
    if not hasattr(g2_orig, 'edge_type'):
        g2_orig = add_edge_types_to_graph(g2_orig)

    edge_type = g1_orig.edge_type.to(device) if hasattr(g1_orig, 'edge_type') else None

    hard_assign, pred_labels, true_labels, misclassified_nodes = get_misclassified_node_and_predictions(
        model, g1_orig, g2_orig, P_gt, device, mean, std
    )

    if not misclassified_nodes:
        print(f"  SKIPPING: No WRONGLY matched S-node found")
        return None, None, None, None, None, None, None, None, None

    gnn_s_node_idx = misclassified_nodes[0]
    gnn_a_node_idx = pred_labels[gnn_s_node_idx].item()

    print(f"    Selected S-node: {gnn_s_node_idx} (valid range: 0-{g2_orig.x.shape[0]-1})")
    print(f"    Selected A-node: {gnn_a_node_idx} (valid range: 0-{g1_orig.x.shape[0]-1})")

    if gnn_s_node_idx >= g2_orig.x.shape[0]:
        print(f"    ERROR: S-node {gnn_s_node_idx} out of range! Max is {g2_orig.x.shape[0]-1}")
        return None, None, None, None, None, None, None, None, None

    if gnn_a_node_idx >= g1_orig.x.shape[0]:
        print(f"    ERROR: A-node {gnn_a_node_idx} out of range! Max is {g1_orig.x.shape[0]-1}")
        return None, None, None, None, None, None, None, None, None

    classifier_pred_class = pred_labels[gnn_s_node_idx].item()
    classifier_true_class = true_labels[gnn_s_node_idx].item()

    print(f"  Found WRONGLY matched GNN S-node: {gnn_s_node_idx}")
    print(f"    GNN predicted: S-{gnn_s_node_idx} → A-{gnn_a_node_idx}")
    print(f"    Ground truth: S-{gnn_s_node_idx} → A-{classifier_true_class}")

    for param in model.encoder.parameters():
        param.requires_grad = False

    uses_edge_type = hasattr(model.encoder, 'edge_embedding')
    classifier = NodeClassifier(model.encoder, P_gt.shape[1], edge_type).to(device)

    x = g1_orig.x.to(device)
    edge_index = g1_orig.edge_index.to(device)
    labels = P_gt.argmax(dim=1).to(device)
    edge_type_for_classifier = edge_type if uses_edge_type else None

    optimizer = torch.optim.Adam(classifier.classifier.parameters(), lr=0.01)
    for epoch in range(epochs):
        classifier.train()
        optimizer.zero_grad()
        if uses_edge_type:
            out = classifier(x, edge_index, edge_attr=edge_type_for_classifier)
        else:
            out = classifier(x, edge_index)
        loss = F.cross_entropy(out, labels)
        loss.backward()
        optimizer.step()

    classifier.eval()
    with torch.no_grad():
        if uses_edge_type:
            logits = classifier(x, edge_index, edge_attr=edge_type_for_classifier)
        else:
            logits = classifier(x, edge_index)
        accuracy = (logits.argmax(dim=1) == labels).float().mean().item()

    explainer = Explainer(
        model=classifier,
        algorithm=GNNExplainer(epochs=100, lr=0.001),
        explanation_type='model',
        model_config=dict(
            mode='multiclass_classification',
            task_level='node',
            return_type='raw',
        ),
        node_mask_type='attributes',
        edge_mask_type='object'
    )

    explanation = explainer(x=x, edge_index=edge_index, edge_attr=edge_type_for_classifier, index=gnn_a_node_idx)

    node_mask = explanation.node_mask.cpu().numpy()
    if node_mask.ndim == 2:
        feature_importance = node_mask[gnn_a_node_idx]
        node_importance = node_mask.mean(axis=1)
    else:
        feature_importance = None
        node_importance = node_mask

    edge_importance = explanation.edge_mask.cpu().numpy()
    if edge_importance.ndim > 1:
        edge_importance = edge_importance.mean(axis=1)

    feature_names = ['Type_Room', 'Type_WS', 'Centroid_X', 'Centroid_Y', 'Normal_X', 'Normal_Y', 'Segment_Length']
    if feature_importance is not None:
        print(f"\n  FEATURE IMPORTANCE for A-node {gnn_a_node_idx}:")
        for f_idx, (f_name, imp) in enumerate(zip(feature_names, feature_importance)):
            print(f"    {f_name:15s}: {imp:.4f}")

    print(f"\n  EDGE IMPORTANCE:")
    print(f"    Min: {edge_importance.min():.4f}")
    print(f"    Max: {edge_importance.max():.4f}")
    print(f"    Mean: {edge_importance.mean():.4f}")
    print(f"    Std: {edge_importance.std():.4f}")

    if edge_type is not None:
        edge_type_np = edge_type.cpu().numpy()
        edge_imp_np = edge_importance

        edge_type_names = {
            0: "Room↔WS",
            1: "Room↔Room",
            2: "WS↔WS"
        }

        print(f"\n  EDGE TYPE IMPORTANCE BREAKDOWN:")
        for e_type in range(3):
            mask = (edge_type_np == e_type)
            if mask.sum() > 0:
                type_imp = edge_imp_np[mask]
                print(f"    {edge_type_names[e_type]}:")
                print(f"      mean={type_imp.mean():.4f}, std={type_imp.std():.4f}, count={mask.sum()}")
                print(f"      min={type_imp.min():.4f}, max={type_imp.max():.4f}")
                print(f"      median={np.median(type_imp):.4f}")
                print(f"      >0.1: {(type_imp > 0.1).sum()}, >0.5: {(type_imp > 0.5).sum()}")

    for param in model.encoder.parameters():
        param.requires_grad = True

    return node_importance, edge_importance, feature_importance, gnn_a_node_idx, gnn_s_node_idx, classifier_pred_class, classifier_true_class, accuracy, hard_assign


def get_gnnexplainer_importance_for_correct(model, g1_orig, g2_orig, P_gt, device, mean, std, epochs=100):
    """
    Use GNNExplainer to explain a CORRECTLY matched S-node.
    """
    print(f"\n  [DEBUG] get_gnnexplainer_importance_for_correct:")
    print(f"    S-graph has {g2_orig.x.shape[0]} nodes")
    print(f"    A-graph has {g1_orig.x.shape[0]} nodes")

    if not hasattr(g1_orig, 'edge_type'):
        g1_orig = add_edge_types_to_graph(g1_orig)
    if not hasattr(g2_orig, 'edge_type'):
        g2_orig = add_edge_types_to_graph(g2_orig)

    edge_type = g1_orig.edge_type.to(device) if hasattr(g1_orig, 'edge_type') else None

    hard_assign, pred_labels, true_labels, misclassified_nodes = get_misclassified_node_and_predictions(
        model, g1_orig, g2_orig, P_gt, device, mean, std
    )

    gnn_s_node_idx = None
    gnn_a_node_idx = None

    for s_idx in range(hard_assign.shape[0]):
        if s_idx not in misclassified_nodes:
            row = hard_assign[s_idx]
            if row.sum().item() > 0:
                predicted_a = row.argmax().item()
                if predicted_a < P_gt.shape[0] and s_idx < P_gt.shape[1]:
                    if P_gt[predicted_a, s_idx] == 1:
                        gnn_s_node_idx = s_idx
                        gnn_a_node_idx = predicted_a
                        break

    if gnn_s_node_idx is None:
        print(f"  SKIPPING: No CORRECTLY matched S-node found")
        return None, None, None, None, None, None, None, None, None

    print(f"    Selected S-node: {gnn_s_node_idx} (valid range: 0-{g2_orig.x.shape[0]-1})")
    print(f"    Selected A-node: {gnn_a_node_idx} (valid range: 0-{g1_orig.x.shape[0]-1})")

    classifier_pred_class = pred_labels[gnn_s_node_idx].item()
    classifier_true_class = true_labels[gnn_s_node_idx].item()

    print(f"  Found CORRECTLY matched GNN S-node: {gnn_s_node_idx}")
    print(f"    GNN predicted: S-{gnn_s_node_idx} → A-{gnn_a_node_idx}")
    print(f"    Ground truth: S-{gnn_s_node_idx} → A-{classifier_true_class}")

    for param in model.encoder.parameters():
        param.requires_grad = False

    uses_edge_type = hasattr(model.encoder, 'edge_embedding')
    classifier = NodeClassifier(model.encoder, P_gt.shape[1], edge_type).to(device)

    x = g1_orig.x.to(device)
    edge_index = g1_orig.edge_index.to(device)
    labels = P_gt.argmax(dim=1).to(device)
    edge_type_for_classifier = edge_type if uses_edge_type else None

    optimizer = torch.optim.Adam(classifier.classifier.parameters(), lr=0.01)
    for epoch in range(epochs):
        classifier.train()
        optimizer.zero_grad()
        if uses_edge_type:
            out = classifier(x, edge_index, edge_attr=edge_type_for_classifier)
        else:
            out = classifier(x, edge_index)
        loss = F.cross_entropy(out, labels)
        loss.backward()
        optimizer.step()

    classifier.eval()
    with torch.no_grad():
        if uses_edge_type:
            logits = classifier(x, edge_index, edge_attr=edge_type_for_classifier)
        else:
            logits = classifier(x, edge_index)
        accuracy = (logits.argmax(dim=1) == labels).float().mean().item()

    explainer = Explainer(
        model=classifier,
        algorithm=GNNExplainer(epochs=100, lr=0.001),
        explanation_type='model',
        model_config=dict(
            mode='multiclass_classification',
            task_level='node',
            return_type='raw',
        ),
        node_mask_type='attributes',
        edge_mask_type='object'
    )

    explanation = explainer(x=x, edge_index=edge_index, edge_attr=edge_type_for_classifier, index=gnn_a_node_idx)

    node_mask = explanation.node_mask.cpu().numpy()
    if node_mask.ndim == 2:
        feature_importance = node_mask[gnn_a_node_idx]
        node_importance = node_mask.mean(axis=1)
    else:
        feature_importance = None
        node_importance = node_mask

    edge_importance = explanation.edge_mask.cpu().numpy()
    if edge_importance.ndim > 1:
        edge_importance = edge_importance.mean(axis=1)

    feature_names = ['Type_Room', 'Type_WS', 'Centroid_X', 'Centroid_Y', 'Normal_X', 'Normal_Y', 'Segment_Length']
    if feature_importance is not None:
        print(f"\n  FEATURE IMPORTANCE for A-node {gnn_a_node_idx}:")
        for f_idx, (f_name, imp) in enumerate(zip(feature_names, feature_importance)):
            print(f"    {f_name:15s}: {imp:.4f}")

    print(f"\n  EDGE IMPORTANCE:")
    print(f"    Min: {edge_importance.min():.4f}")
    print(f"    Max: {edge_importance.max():.4f}")
    print(f"    Mean: {edge_importance.mean():.4f}")
    print(f"    Std: {edge_importance.std():.4f}")

    if edge_type is not None:
        edge_type_np = edge_type.cpu().numpy()
        edge_imp_np = edge_importance

        edge_type_names = {
            0: "Room↔WS",
            1: "Room↔Room",
            2: "WS↔WS"
        }

        print(f"\n  EDGE TYPE IMPORTANCE BREAKDOWN:")
        for e_type in range(3):
            mask = (edge_type_np == e_type)
            if mask.sum() > 0:
                type_imp = edge_imp_np[mask]
                print(f"    {edge_type_names[e_type]}:")
                print(f"      mean={type_imp.mean():.4f}, std={type_imp.std():.4f}, count={mask.sum()}")
                print(f"      min={type_imp.min():.4f}, max={type_imp.max():.4f}")
                print(f"      median={np.median(type_imp):.4f}")
                print(f"      >0.1: {(type_imp > 0.1).sum()}, >0.5: {(type_imp > 0.5).sum()}")

    for param in model.encoder.parameters():
        param.requires_grad = True

    return node_importance, edge_importance, feature_importance, gnn_a_node_idx, gnn_s_node_idx, classifier_pred_class, classifier_true_class, accuracy, hard_assign