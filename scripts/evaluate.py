import os
import argparse
import pickle
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

from config import ModelParams
from data import (
    GraphMatchingDataset,
    deserialize_graph_matching_dataset,
    compute_mean_std,
    normalize_data_pairs,
    collate_pyg_matching,
    add_edge_types_to_graph,
    plot_two_graphs_with_matching,
)
from models import (
    GCNEncoder,
    GraphSAGEEncoder,
    GINEncoder,
    GATv2Encoder,
    GraphTransformerEncoder,
    GINEEncoder,
    RGCNEncoder,
    GraphMatcher,
)
from utils import set_seed, evaluate, evaluate_all_test_samples, measure_inference_time


def get_model(model_name, params):
    """Get model by name."""
    if model_name == "gatv2":
        encoder = GATv2Encoder(params)
        return GraphMatcher(encoder, params)
    elif model_name == "gcn":
        encoder = GCNEncoder(params)
        return GraphMatcher(encoder, params)
    elif model_name == "sage":
        encoder = GraphSAGEEncoder(params)
        return GraphMatcher(encoder, params)
    elif model_name == "gin":
        encoder = GINEncoder(params)
        return GraphMatcher(encoder, params)
    elif model_name == "gine":
        encoder = GINEEncoder(params)
        return GraphMatcher(encoder, params)
    elif model_name == "rgcn":
        encoder = RGCNEncoder(params)
        return GraphMatcher(encoder, params)
    elif model_name == "transformer":
        encoder = GraphTransformerEncoder(params)
        return GraphMatcher(encoder, params)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def main(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set seed
    set_seed(args.seed)

    # Load data
    data_path = args.data_path
    train_pairs = deserialize_graph_matching_dataset(data_path, "train_dataset.pkl")
    val_pairs = deserialize_graph_matching_dataset(data_path, "valid_dataset.pkl")
    test_pairs = deserialize_graph_matching_dataset(data_path, "test_dataset.pkl")

    # Load original and noise graphs for visualization
    with open(os.path.join(data_path, "original.pkl"), 'rb') as f:
        original_graphs_nx = pickle.load(f)
    with open(os.path.join(data_path, "noise.pkl"), 'rb') as f:
        noise_graphs_nx = pickle.load(f)

    # Compute normalization statistics
    mean, std = compute_mean_std(train_pairs)

    # Normalize datasets
    train_pairs_norm = normalize_data_pairs(train_pairs, mean, std)
    val_pairs_norm = normalize_data_pairs(val_pairs, mean, std)
    test_pairs_norm = normalize_data_pairs(test_pairs, mean, std)

    # Add edge types for models that need them
    if args.model in ["transformer", "gine", "rgcn"]:
        print("Adding edge types to datasets...")
        test_pairs_norm = [(add_edge_types_to_graph(g1), add_edge_types_to_graph(g2), P) 
                           for g1, g2, P in test_pairs_norm]

    # Create test dataset and dataloader
    test_dataset = GraphMatchingDataset(test_pairs_norm)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_pyg_matching
    )

    # Load model
    params = ModelParams.from_json(args.params_file) if args.params_file else ModelParams.get_default()
    model = get_model(args.model, params).to(device)

    checkpoint_path = Path(args.checkpoint_dir) / f"best_{args.model}_model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from {checkpoint_path}")

    # Evaluate
    print("\nTEST SET EVALUATION")
    test_metrics, test_detailed = evaluate(model, test_loader, device)

    print("TEST SET RESULTS")
    print(f"{'Metric':<15} {'Value':<15}")
    print(f"{'Precision':<15} {test_metrics['precision']:.4f}")
    print(f"{'Recall':<15} {test_metrics['recall']:.4f}")
    print(f"{'F1 Score':<15} {test_metrics['f1']:.4f} ← Primary")
    print(f"{'Accuracy':<15} {test_metrics['accuracy']:.4f}")

    # F1 score distribution
    f1_scores = [m['f1'] for m in test_detailed]
    print("\nF1 SCORE DISTRIBUTION ANALYSIS")
    print(f"   Mean: {np.mean(f1_scores):.4f}")
    print(f"   Median: {np.median(f1_scores):.4f}")
    print(f"   Std: {np.std(f1_scores):.4f}")
    print(f"   Min: {np.min(f1_scores):.4f}")
    print(f"   Max: {np.max(f1_scores):.4f}")
    print(f"   25th percentile: {np.percentile(f1_scores, 25):.4f}")
    print(f"   75th percentile: {np.percentile(f1_scores, 75):.4f}")

    # Measure inference time
    avg_time, std_time = measure_inference_time(
        model, test_loader, device, 
        num_samples=min(100, len(test_dataset)), warmup=20
    )
    print(f"\nAverage inference time: {avg_time:.2f} ± {std_time:.1f} ms")

    # Visualize predictions
    if args.visualize:
        print("\nVisualizing test samples...")
        num_samples = min(args.num_vis_samples, len(test_pairs))
        vis_dir = Path(args.output_dir)
        vis_dir.mkdir(parents=True, exist_ok=True)

        for idx in range(num_samples):
            g1_orig, g2_orig, P_gt = test_pairs[idx]

            # Normalize on the fly
            g1_norm = Data(x=(g1_orig.x - mean) / (std + 1e-8), edge_index=g1_orig.edge_index)
            g2_norm = Data(x=(g2_orig.x - mean) / (std + 1e-8), edge_index=g2_orig.edge_index)

            if hasattr(g1_orig, 'edge_type'):
                g1_norm.edge_type = g1_orig.edge_type
                g2_norm.edge_type = g2_orig.edge_type

            batch1 = Batch.from_data_list([g1_norm]).to(device)
            batch2 = Batch.from_data_list([g2_norm]).to(device)

            with torch.no_grad():
                S_pred_list, _ = model(batch1, batch2)
                S_pred = S_pred_list[0]

            N_A, N_S = P_gt.shape
            S_real = S_pred[:, :N_S]

            try:
                hard_assign_a_to_s = pygmtools.hungarian(
                    S_real.unsqueeze(0),
                    n1=torch.tensor([N_A]),
                    n2=torch.tensor([N_S])
                ).squeeze(0)
            except:
                max_dim = max(N_A, N_S)
                S_padded = torch.zeros(max_dim, max_dim, device=S_real.device)
                S_padded[:N_A, :N_S] = S_real
                if N_A > N_S:
                    S_padded[N_A:, :N_S] = 1e-9
                    S_padded[:N_A, N_S:] = 1e-9
                hard_assign_full = pygmtools.hungarian(S_padded.unsqueeze(0)).squeeze(0)
                hard_assign_a_to_s = hard_assign_full[:N_A, :N_S]

            hard_assign_s_to_a = hard_assign_a_to_s.T
            P_gt_s_to_a = P_gt.T

            # Compute metrics
            tp = ((hard_assign_a_to_s == 1) & (P_gt == 1)).sum().item()
            fp = ((hard_assign_a_to_s == 1) & (P_gt == 0)).sum().item()
            fn = ((hard_assign_a_to_s == 0) & (P_gt == 1)).sum().item()
            tn = ((hard_assign_a_to_s == 0) & (P_gt == 0)).sum().item()

            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)

            # Create visualization
            fig, ax = plot_two_graphs_with_matching(
                graphs_list=[g1_orig, g2_orig],
                gt_perm=P_gt_s_to_a.cpu(),
                original_graphs=original_graphs_nx,
                pred_perm=hard_assign_s_to_a.cpu(),
                noise_graphs=noise_graphs_nx,
                viz_rooms=True,
                viz_ws=True,
                viz_room_connection=True,
                viz_normals=False,
                viz_room_normals=False,
                match_display="all",
                title=f"Test Sample {idx+1}: Predictions (F1={f1:.3f}) | ✓{tp} Correct, ✗{fp} Wrong",
                save_path=str(vis_dir / f"test_sample_{idx+1}.png")
            )
            plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate graph matching models")
    parser.add_argument("--model", type=str, default="gatv2",
                        choices=["gatv2", "gcn", "sage", "gin", "gine", "rgcn", "transformer"],
                        help="Model architecture to use")
    parser.add_argument("--data_path", type=str, default="/content/msd_data",
                        help="Path to dataset")
    parser.add_argument("--params_file", type=str, default=None,
                        help="Path to JSON parameters file")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints",
                        help="Directory containing checkpoints")
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Directory for output results")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for evaluation")
    parser.add_argument("--visualize", action="store_true",
                        help="Visualize predictions on test samples")
    parser.add_argument("--num_vis_samples", type=int, default=10,
                        help="Number of samples to visualize")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    from torch_geometric.data import Data
    import pygmtools
    pygmtools.BACKEND = 'pytorch'

    main(args)