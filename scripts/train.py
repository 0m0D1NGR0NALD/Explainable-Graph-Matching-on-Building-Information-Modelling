import os
import argparse
import pickle
from pathlib import Path

import torch
import torch.nn as nn
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
    MatchingModel_MLPGATv2Sinkhorn,
)
from utils import set_seed, evaluate, measure_inference_time


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

    # Load parameters
    if args.params_file:
        params = ModelParams.from_json(args.params_file)
    else:
        params = ModelParams.get_default()

    print(f"Using parameters: {params}")

    # Load data
    data_path = args.data_path
    train_pairs = deserialize_graph_matching_dataset(data_path, "train_dataset.pkl")
    val_pairs = deserialize_graph_matching_dataset(data_path, "valid_dataset.pkl")
    test_pairs = deserialize_graph_matching_dataset(data_path, "test_dataset.pkl")

    print(f"Train: {len(train_pairs)}, Val: {len(val_pairs)}, Test: {len(test_pairs)}")

    # Compute normalization statistics
    mean, std = compute_mean_std(train_pairs)

    # Normalize datasets
    train_pairs_norm = normalize_data_pairs(train_pairs, mean, std)
    val_pairs_norm = normalize_data_pairs(val_pairs, mean, std)
    test_pairs_norm = normalize_data_pairs(test_pairs, mean, std)

    # Add edge types for models that need them
    if args.model in ["transformer", "gine", "rgcn"]:
        print("Adding edge types to datasets...")
        train_pairs_norm = [(add_edge_types_to_graph(g1), add_edge_types_to_graph(g2), P) 
                            for g1, g2, P in train_pairs_norm]
        val_pairs_norm = [(add_edge_types_to_graph(g1), add_edge_types_to_graph(g2), P) 
                          for g1, g2, P in val_pairs_norm]
        test_pairs_norm = [(add_edge_types_to_graph(g1), add_edge_types_to_graph(g2), P) 
                           for g1, g2, P in test_pairs_norm]

    # Create datasets and dataloaders
    train_dataset = GraphMatchingDataset(train_pairs_norm)
    val_dataset = GraphMatchingDataset(val_pairs_norm)
    test_dataset = GraphMatchingDataset(test_pairs_norm)

    train_loader = DataLoader(
        train_dataset, batch_size=params.batch_size, shuffle=True, collate_fn=collate_pyg_matching
    )
    val_loader = DataLoader(
        val_dataset, batch_size=params.batch_size, shuffle=False, collate_fn=collate_pyg_matching
    )
    test_loader = DataLoader(
        test_dataset, batch_size=params.batch_size, shuffle=False, collate_fn=collate_pyg_matching
    )

    # Initialize model
    model = get_model(args.model, params).to(device)

    print(f"Model: {args.model}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=params.learning_rate,
        weight_decay=params.weight_decay
    )

    # Training
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    val_f1_scores = []

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"best_{args.model}_model.pt"

    # Resume training if checkpoint exists
    start_epoch = 0
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        train_losses = checkpoint.get('train_losses', [])
        val_losses = checkpoint.get('val_losses', [])
        val_f1_scores = checkpoint.get('val_f1_scores', [])
        print(f"Resuming from epoch {start_epoch}, best val loss: {best_val_loss:.4f}")

    print(f"\nStarting training for {params.num_epochs} epochs...")

    for epoch in range(start_epoch, params.num_epochs):
        # Training
        model.train()
        epoch_train_loss = 0
        num_batches = 0

        for batch1, batch2, perm_list in train_loader:
            batch1 = batch1.to(device)
            batch2 = batch2.to(device)
            perm_list = [p.to(device) for p in perm_list]

            S_pred_list, _ = model(batch1, batch2)

            batch_loss = 0
            for i, S_pred in enumerate(S_pred_list):
                P_gt = perm_list[i]
                loss = permutation_loss(S_pred, P_gt)
                batch_loss += loss

            batch_loss = batch_loss / len(S_pred_list)

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            epoch_train_loss += batch_loss.item()
            num_batches += 1

        avg_train_loss = epoch_train_loss / num_batches if num_batches > 0 else 0
        train_losses.append(avg_train_loss)

        # Validation
        val_metrics, _ = evaluate(model, val_loader, device, verbose=False)
        val_loss = val_metrics.get('loss', 0)
        val_losses.append(val_loss)
        val_f1_scores.append(val_metrics['f1'])

        print(f"Epoch {epoch+1:3d}/{params.num_epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val F1: {val_metrics['f1']:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'val_f1_scores': val_f1_scores,
                'hyperparams': params.to_dict()
            }, checkpoint_path)
            print(f"  → New best model! Val Loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= params.patience:
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                break

    # Load best model for evaluation
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    print("\nTRAINING COMPLETE")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Total epochs trained: {len(train_losses)}")

    # Test evaluation
    print("\nTEST SET EVALUATION")
    test_metrics, test_detailed = evaluate(model, test_loader, device)

    print("TEST SET RESULTS")
    print(f"{'Metric':<15} {'Value':<15}")
    print(f"{'Precision':<15} {test_metrics['precision']:.4f}")
    print(f"{'Recall':<15} {test_metrics['recall']:.4f}")
    print(f"{'F1 Score':<15} {test_metrics['f1']:.4f} ← Primary")
    print(f"{'Accuracy':<15} {test_metrics['accuracy']:.4f}")

    # Measure inference time
    avg_time, std_time = measure_inference_time(
        model, test_loader, device, 
        num_samples=min(100, len(test_dataset)), warmup=20
    )
    print(f"\nAverage inference time: {avg_time:.2f} ± {std_time:.1f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train graph matching models")
    parser.add_argument("--model", type=str, default="gatv2",
                        choices=["gatv2", "gcn", "sage", "gin", "gine", "rgcn", "transformer"],
                        help="Model architecture to use")
    parser.add_argument("--data_path", type=str, default="/content/msd_data",
                        help="Path to dataset")
    parser.add_argument("--params_file", type=str, default=None,
                        help="Path to JSON parameters file")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    main(args)