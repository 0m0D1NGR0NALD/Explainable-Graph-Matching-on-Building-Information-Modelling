import time
import numpy as np
from tqdm import tqdm
import torch

from .metrics import compute_metrics, permutation_loss


def evaluate(model, loader, device, verbose=True):
    """
    Evaluate model and return metrics.
    """
    model.eval()
    total_loss = 0
    all_metrics = []

    iterator = tqdm(loader, desc="Evaluating") if verbose else loader

    with torch.no_grad():
        for batch1, batch2, perm_list in iterator:
            batch1 = batch1.to(device)
            batch2 = batch2.to(device)
            perm_list = [p.to(device) for p in perm_list]

            S_pred_list, _ = model(batch1, batch2)

            for i, S_pred in enumerate(S_pred_list):
                P_gt = perm_list[i]
                loss = permutation_loss(S_pred, P_gt)
                metrics = compute_metrics(S_pred, P_gt)
                metrics['loss'] = loss.item()

                all_metrics.append(metrics)
                total_loss += loss.item()

    avg_metrics = {
        'loss': total_loss / len(all_metrics),
        'precision': np.mean([m['precision'] for m in all_metrics]),
        'recall': np.mean([m['recall'] for m in all_metrics]),
        'f1': np.mean([m['f1'] for m in all_metrics]),
        'accuracy': np.mean([m['accuracy'] for m in all_metrics])
    }

    return avg_metrics, all_metrics


def evaluate_all_test_samples(model, test_loader, device):
    """
    Evaluate all test samples individually
    """
    model.eval()
    sample_results = []

    with torch.no_grad():
        sample_idx = 0
        for batch1, batch2, perm_list in tqdm(test_loader):
            batch1 = batch1.to(device)
            batch2 = batch2.to(device)
            perm_list = [p.to(device) for p in perm_list]

            S_pred_list, _ = model(batch1, batch2)

            for i, S_pred in enumerate(S_pred_list):
                P_gt = perm_list[i]

                metrics = compute_metrics(S_pred, P_gt)

                sample_results.append({
                    'sample_idx': sample_idx,
                    'n1': P_gt.shape[0],
                    'n2': P_gt.shape[1],
                    'f1': metrics['f1'],
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'accuracy': metrics['accuracy'],
                    'tp': metrics['tp'],
                    'fp': metrics['fp'],
                    'fn': metrics['fn'],
                    'tn': metrics['tn']
                })
                sample_idx += 1

    return sample_results


def measure_inference_time(model, test_loader, device, num_samples=100, warmup=20):
    """
    Measure inference time for the model.
    """
    model.eval()
    inference_times = []

    print(f"Warm-up samples: {warmup}")
    print(f"Measurement samples: {num_samples}")

    with torch.no_grad():
        samples_warmed = 0
        samples_measured = 0

        for batch1, batch2, perm_list in test_loader:
            batch1 = batch1.to(device)
            batch2 = batch2.to(device)

            num_graphs = batch1.num_graphs if hasattr(batch1, 'num_graphs') else len(perm_list)

            for i in range(num_graphs):
                if samples_warmed < warmup:
                    S_pred_list, _ = model(batch1, batch2)
                    _ = S_pred_list[i]
                    samples_warmed += 1
                    continue

                if samples_measured >= num_samples:
                    break

                if device.type == 'cuda':
                    torch.cuda.synchronize()

                start_time = time.perf_counter()
                S_pred_list, _ = model(batch1, batch2)
                _ = S_pred_list[i]
                if device.type == 'cuda':
                    torch.cuda.synchronize()

                inference_times.append(time.perf_counter() - start_time)
                samples_measured += 1

            if samples_measured >= num_samples:
                break

    avg_time = np.mean(inference_times) * 1000
    std_time = np.std(inference_times) * 1000

    return avg_time, std_time