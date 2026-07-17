import os
import argparse
import pickle
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

from config import ModelParams
from data import (
    deserialize_graph_matching_dataset,
    compute_mean_std,
    normalize_data_pairs,
    collate_pyg_matching,
    add_edge_types_to_graph,
)
from models import (
    GATv2Encoder,
    GraphSAGEEncoder,
    GINEEncoder,
    GraphMatcher,
)
from explain import (
    get_misclassified_node_and_predictions,
    get_gnnexplainer_importance_for_misclassified,
    get_gnnexplainer_importance_for_correct,
    plot_node_importance_with_graphs,
    plot_edge_importance_with_graphs,
)
from utils import set_seed


def get_model(model_name, params):
    """Get model by name."""
    if model_name == "gatv2":
        encoder = GATv2Encoder(params)
        return GraphMatcher(encoder, params)
    elif model_name == "sage":
        encoder = GraphSAGEEncoder(params)
        return GraphMatcher(encoder, params)
    elif model_name == "gine":
        encoder = GINEEncoder(params)
        return GraphMatcher(encoder, params)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    set_seed(args.seed)

    # Load data
    data_path = args.data_path
    train_pairs = deserialize_graph_matching_dataset(data_path, "train_dataset.pkl")
    test_pairs = deserialize_graph_matching_dataset(data_path, "test_dataset.pkl")
    
    with open(os.path.join(data_path, "original.pkl"), 'rb') as f:
        original_graphs_nx = pickle.load(f)
    with open(os.path.join(data_path, "noise.pkl"), 'rb') as f:
        noise_graphs_nx = pickle.load(f)

    # Compute normalization statistics
    mean, std = compute_mean_std(train_pairs)

    # Load model
    params = ModelParams.get_default()
    model = get_model(args.model, params).to(device)
    
    checkpoint_path = Path(args.checkpoint_dir) / f"best_{args.model}_model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from {checkpoint_path}")

    # Add edge types if needed
    if args.model in ["gine"]:
        test_pairs = [(add_edge_types_to_graph(g1), add_edge_types_to_graph(g2), P) 
                      for g1, g2, P in test_pairs]

    # Analyze samples
    print(f"\nAnalyzing {args.num_samples} test samples...")
    
    for idx in range(min(args.num_samples, len(test_pairs))):
        g1_orig, g2_orig, P_gt = test_pairs[idx]
        
        print(f"\n{'='*60}")
        print(f"SAMPLE {idx+1}: A-nodes={g1_orig.x.shape[0]}, S-nodes={g2_orig.x.shape[0]}")
        print(f"{'='*60}")
        
        # Get predictions
        hard_assign, pred_labels, true_labels, misclassified_nodes = get_misclassified_node_and_predictions(
            model, g1_orig, g2_orig, P_gt, device, mean, std
        )
        
        # Analyze misclassified nodes
        if misclassified_nodes and args.explain_wrong:
            print(f"\nAnalyzing misclassified nodes: {misclassified_nodes[:5]}...")
            
            result = get_gnnexplainer_importance_for_misclassified(
                model, g1_orig, g2_orig, P_gt.cpu(), device, mean, std, epochs=args.epochs
            )
            
            if result[0] is not None:
                node_importance, edge_importance, feature_importance, a_node_idx, s_node_idx, pred_class, true_class, accuracy, hard_assign = result
                
                # Get node names
                s_graph_nodes_permuted = [g2_orig.node_names[i] for i in range(g2_orig.x.shape[0])]
                target_s_node_name = s_graph_nodes_permuted[s_node_idx]
                gt_perm_transposed = P_gt.cpu().T
                
                # Visualize
                if args.visualize:
                    output_dir = Path(args.output_dir) / f"sample_{idx+1}"
                    output_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Node importance plot
                    fig, ax = plot_node_importance_with_graphs(
                        graphs_list=[g1_orig, g2_orig],
                        gt_perm=gt_perm_transposed,
                        original_graphs=original_graphs_nx,
                        node_importance=node_importance,
                        target_node_id=target_s_node_name,
                        target_s_node_idx=s_node_idx,
                        pred_perm=hard_assign,
                        noise_graphs=noise_graphs_nx,
                        title=f"Sample {idx+1} | S-node {s_node_idx} (GT: A-{true_class}) -> pred A-{a_node_idx} | NODE IMPORTANCE",
                        show_all_matches=False,
                        save_path=str(output_dir / "node_importance.png")
                    )
                    plt.close(fig)
                    
                    # Edge importance plot
                    fig, ax = plot_edge_importance_with_graphs(
                        graphs_list=[g1_orig, g2_orig],
                        gt_perm=gt_perm_transposed,
                        original_graphs=original_graphs_nx,
                        edge_importance=edge_importance,
                        target_node_id=target_s_node_name,
                        target_s_node_idx=s_node_idx,
                        pred_perm=hard_assign,
                        noise_graphs=noise_graphs_nx,
                        title=f"Sample {idx+1} | S-node {s_node_idx} (GT: A-{true_class}) -> pred A-{a_node_idx} | EDGE IMPORTANCE",
                        save_path=str(output_dir / "edge_importance.png")
                    )
                    plt.close(fig)
        
        # Analyze correct nodes
        if args.explain_correct:
            correct_nodes = [j for j in range(len(true_labels)) if j not in misclassified_nodes]
            if correct_nodes:
                print(f"\nAnalyzing correct nodes: {correct_nodes[:5]}...")
                
                result = get_gnnexplainer_importance_for_correct(
                    model, g1_orig, g2_orig, P_gt.cpu(), device, mean, std, epochs=args.epochs
                )

    print("\nAnalysis complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Explain graph matching model predictions")
    parser.add_argument("--model", type=str, default="gatv2",
                        choices=["gatv2", "sage", "gine"],
                        help="Model architecture to explain")
    parser.add_argument("--data_path", type=str, default="/content/msd_data",
                        help="Path to dataset")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints",
                        help="Directory containing checkpoints")
    parser.add_argument("--output_dir", type=str, default="./explanations",
                        help="Directory for output explanations")
    parser.add_argument("--num_samples", type=int, default=5,
                        help="Number of samples to analyze")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of epochs for GNNExplainer")
    parser.add_argument("--explain_wrong", action="store_true",
                        help="Explain misclassified nodes")
    parser.add_argument("--explain_correct", action="store_true",
                        help="Explain correctly classified nodes")
    parser.add_argument("--visualize", action="store_true",
                        help="Generate visualization plots")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()
    
    main(args)