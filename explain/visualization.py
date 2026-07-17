import copy
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import networkx as nx
from shapely.geometry import Polygon
from shapely.affinity import translate

from data import pyg_data_to_nx_digraph


def plot_node_importance_with_graphs(graphs_list, gt_perm, original_graphs,
                                      node_importance, target_node_id, target_s_node_idx,
                                      pred_perm=None, noise_graphs=None,
                                      title=None, save_path=None,
                                      show_all_matches=False):
    """
    Node importance visualization with both A-graph and S-graph side by side.
    """
    if noise_graphs is None:
        noise_graphs = original_graphs

    g1tensor, g2tensor = copy.deepcopy(graphs_list[0]), copy.deepcopy(graphs_list[1])

    if g2tensor.x.shape[0] == 0:
        print(f"  WARNING: S-graph has no nodes, skipping visualization")
        return None, None

    node_names_a = list(g1tensor.node_names)
    orig_names_s = list(g2tensor.node_names)
    perm = g2tensor.permutation.tolist()
    node_names_s = [orig_names_s[p] for p in perm]

    try:
        g_a = pyg_data_to_nx_digraph(g1tensor, original_graphs)
        g_s_original = pyg_data_to_nx_digraph(g2tensor, noise_graphs)
        g_s = g_s_original.copy()
    except ValueError as e:
        print(f"  WARNING: Could not convert graph: {e}")
        return None, None

    if len(g_s.nodes()) == 0:
        print(f"  WARNING: S-graph has no nodes after conversion, skipping visualization")
        return None, None

    matched_s_nodes = set()

    if pred_perm is not None:
        for s_idx in range(min(pred_perm.shape[0], len(node_names_s))):
            row = pred_perm[s_idx]
            if row.sum().item() > 0:
                matched_s_nodes.add(node_names_s[s_idx])

    target_is_matched = target_node_id in matched_s_nodes if target_node_id else False

    if node_importance.max() > node_importance.min():
        node_imp_norm = (node_importance - node_importance.min()) / (node_importance.max() - node_importance.min())
    else:
        node_imp_norm = node_importance

    node_list = list(g_a.nodes())
    node_to_importance = {}
    for i, n in enumerate(node_list):
        node_to_importance[n] = node_imp_norm[i] if i < len(node_imp_norm) else 0

    max_x_a = max(data['center'][0] for _, data in g_a.nodes(data=True))
    min_x_s = min(data['center'][0] for _, data in g_s.nodes(data=True))
    translation_x = (max_x_a - min_x_s) + 10.0
    for _, data in g_s.nodes(data=True):
        data['center'][0] += translation_x
        if 'polygon' in data:
            poly = data['polygon']
            if isinstance(poly, Polygon):
                data['polygon'] = translate(poly, xoff=translation_x)
            else:
                data['polygon'] = Polygon([(x + translation_x, y) for x, y in poly])
        if 'limits' in data:
            data['limits'] = [[x + translation_x, y] for x, y in data['limits']]

    fig, ax = plt.subplots(figsize=(22, 12))
    legend_added = set()

    # A-GRAPH (BIM) - LEFT SIDE
    for u, v in g_a.edges():
        ax.plot([g_a.nodes[u]['center'][0], g_a.nodes[v]['center'][0]],
               [g_a.nodes[u]['center'][1], g_a.nodes[v]['center'][1]],
               color='lightgray', linewidth=1.0, alpha=0.5, zorder=1)

    for n, d in g_a.nodes(data=True):
        if d['type'] == 'room' and 'polygon' in d:
            poly = Polygon(d['polygon']) if not isinstance(d['polygon'], Polygon) else d['polygon']
            x, y = poly.exterior.xy
            ax.fill(x, y, alpha=0.15, fc='lightgray', ec='gray', linewidth=0.8, zorder=2)

    for n, d in g_a.nodes(data=True):
        if d['type'] == 'ws':
            imp = node_to_importance.get(n, 0)
            node_color = plt.cm.RdYlBu_r(imp)
            ax.scatter(d['center'][0], d['center'][1],
                      color=node_color, s=100,
                      edgecolors='black', linewidth=1.5, zorder=5)
            if 'limits' in d:
                l1, l2 = d['limits']
                ax.plot([l1[0], l2[0]], [l1[1], l2[1]], 'gray', linewidth=1.0, alpha=0.6)

    for n, d in g_a.nodes(data=True):
        if d['type'] == 'room':
            center = d['center']
            imp = node_to_importance.get(n, 0)
            node_color = plt.cm.RdYlBu_r(imp)
            ax.scatter(center[0], center[1], color=node_color, s=120,
                      edgecolors='black', linewidth=2, zorder=5)
            ax.annotate(f'{imp:.2f}', (center[0], center[1]),
                       fontsize=7, ha='center', va='center', color='white',
                       bbox=dict(boxstyle='round', facecolor='black', alpha=0.6))

    # S-GRAPH (Robot) - RIGHT SIDE
    for u, v in g_s.edges():
        ax.plot([g_s.nodes[u]['center'][0], g_s.nodes[v]['center'][0]],
               [g_s.nodes[u]['center'][1], g_s.nodes[v]['center'][1]],
               color='lightgray', linewidth=1.0, alpha=0.5, zorder=1)

    for n, d in g_s.nodes(data=True):
        if d['type'] == 'room' and 'polygon' in d:
            poly = Polygon(d['polygon']) if not isinstance(d['polygon'], Polygon) else d['polygon']
            x, y = poly.exterior.xy
            ax.fill(x, y, alpha=0.15, fc='lightcoral', ec='gray', linewidth=0.8, zorder=2)

    for n, d in g_s.nodes(data=True):
        if d['type'] == 'ws':
            is_target = (n == target_node_id)

            if is_target:
                if 'Target S-node' not in legend_added:
                    ax.scatter(d['center'][0], d['center'][1], c='gold', s=120, marker='o',
                              edgecolors='darkorange', linewidth=2.5, zorder=10,
                              label='Target S-node')
                    legend_added.add('Target S-node')
                else:
                    ax.scatter(d['center'][0], d['center'][1], c='gold', s=120, marker='o',
                              edgecolors='darkorange', linewidth=2.5, zorder=10)
            else:
                if 'Matched S-node' not in legend_added:
                    ax.scatter(d['center'][0], d['center'][1], color='mediumpurple', s=50,
                              edgecolors='indigo', linewidth=1.5, zorder=3,
                              label='Matched S-node')
                    legend_added.add('Matched S-node')
                else:
                    ax.scatter(d['center'][0], d['center'][1], color='mediumpurple', s=50,
                              edgecolors='indigo', linewidth=1.5, zorder=3)

            if 'limits' in d:
                l1, l2 = d['limits']
                ax.plot([l1[0], l2[0]], [l1[1], l2[1]], 'gray', linewidth=1.0, alpha=0.6)

    for n, d in g_s.nodes(data=True):
        if d['type'] == 'room':
            is_target = (n == target_node_id)

            if is_target:
                center = d['center']
                if 'Target centroid' not in legend_added:
                    ax.scatter(center[0], center[1], c='gold', s=200, marker='o',
                              edgecolors='darkorange', linewidth=2.5, zorder=10,
                              label='Target centroid')
                    legend_added.add('Target centroid')
                else:
                    ax.scatter(center[0], center[1], c='gold', s=200, marker='o',
                              edgecolors='darkorange', linewidth=2.5, zorder=10)
            else:
                center = d['center']
                if 'Matched centroid' not in legend_added:
                    ax.scatter(center[0], center[1], color='#40E0D0', s=80,
                              edgecolors='#008080', linewidth=1.5, zorder=3,
                              label='Matched centroid')
                    legend_added.add('Matched centroid')
                else:
                    ax.scatter(center[0], center[1], color='#40E0D0', s=80,
                              edgecolors='#008080', linewidth=1.5, zorder=3)

    # MATCHING LINES (S → A)
    if pred_perm is not None:
        correct_label_added = False
        wrong_label_added = False

        n_s_nodes = min(pred_perm.shape[0], len(node_names_s))

        for s_idx in range(n_s_nodes):
            if s_idx >= gt_perm.shape[0]:
                continue
            if gt_perm[s_idx].sum().item() == 0:
                continue

            row = pred_perm[s_idx]
            if row.sum().item() == 0:
                continue

            a_idx = row.argmax().item()

            if s_idx >= len(node_names_s):
                continue
            s_node_name = node_names_s[s_idx]
            if s_node_name not in g_s.nodes:
                continue
            if a_idx >= len(node_names_a):
                continue
            a_node_name = node_names_a[a_idx]
            if a_node_name not in g_a.nodes:
                continue

            is_target = (s_idx == target_s_node_idx)

            if not show_all_matches and not is_target:
                continue

            pt_s = g_s.nodes[s_node_name]['center']
            pt_a = g_a.nodes[a_node_name]['center']

            is_correct = (a_idx < gt_perm.shape[1] and gt_perm[s_idx, a_idx] == 1)

            color = 'green' if is_correct else 'red'

            if is_target:
                linewidth = 4.0
                alpha = 0.9
            else:
                linewidth = 2.0
                alpha = 0.5

            if color == 'green' and not correct_label_added:
                ax.plot([pt_s[0], pt_a[0]], [pt_s[1], pt_a[1]],
                       color=color, linestyle='-', alpha=alpha, linewidth=linewidth,
                       label='Correct match')
                correct_label_added = True
            elif color == 'green':
                ax.plot([pt_s[0], pt_a[0]], [pt_s[1], pt_a[1]],
                       color=color, linestyle='-', alpha=alpha, linewidth=linewidth)
            elif color == 'red' and not wrong_label_added:
                ax.plot([pt_s[0], pt_a[0]], [pt_s[1], pt_a[1]],
                       color=color, linestyle='-', alpha=alpha, linewidth=linewidth,
                       label='Wrong match')
                wrong_label_added = True
            else:
                ax.plot([pt_s[0], pt_a[0]], [pt_s[1], pt_a[1]],
                       color=color, linestyle='-', alpha=alpha, linewidth=linewidth)

    # LEGEND
    legend_handles = [
        plt.Rectangle((0,0),1,1, facecolor='lightgray', alpha=0.5, edgecolor='gray'),
        plt.Rectangle((0,0),1,1, facecolor='lightcoral', alpha=0.5, edgecolor='gray'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gold', markersize=10,
                   markeredgecolor='darkorange', linestyle='None'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='mediumpurple', markersize=8,
                   markeredgecolor='indigo', linestyle='None'),
        plt.Line2D([0], [0], color='green', linewidth=2.5),
        plt.Line2D([0], [0], color='red', linewidth=2.5),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10,
                   markeredgecolor='black', linestyle='None'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10,
                   markeredgecolor='black', linestyle='None'),
    ]
    legend_labels = [
        'A-graph (BIM)',
        'S-graph (Robot)',
        'Target S-node',
        'Matched S-node',
        'Correct match',
        'Wrong match',
        'High importance node (A-graph)',
        'Low importance node (A-graph)',
    ]
    ax.legend(legend_handles, legend_labels, loc='upper right', fontsize=9, framealpha=0.9, ncol=2)

    sm = ScalarMappable(norm=Normalize(0, 1), cmap=plt.cm.RdYlBu_r)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.5)
    cbar.set_label('NODE IMPORTANCE (Red=High, Blue=Low)', fontsize=12, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)

    if show_all_matches:
        mode_suffix = " (showing ALL matches)"
    else:
        mode_suffix = " (showing ONLY target match)"
    ax.set_title(f"{title}{mode_suffix}", fontsize=14, fontweight='bold')
    ax.set_xlabel('X (meters)', fontsize=12)
    ax.set_ylabel('Y (meters)', fontsize=12)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.tight_layout()
    return fig, ax


def plot_edge_importance_with_graphs(graphs_list, gt_perm, original_graphs,
                                      edge_importance, target_node_id, target_s_node_idx,
                                      pred_perm=None, noise_graphs=None,
                                      title=None, save_path=None,
                                      show_all_matches=False):
    """
    Edge importance visualization with both A-graph and S-graph side by side.
    """
    if noise_graphs is None:
        noise_graphs = original_graphs

    g1tensor, g2tensor = copy.deepcopy(graphs_list[0]), copy.deepcopy(graphs_list[1])

    if g2tensor.x.shape[0] == 0:
        print(f"  WARNING: S-graph has no nodes, skipping visualization")
        return None, None

    node_names_a = list(g1tensor.node_names)
    orig_names_s = list(g2tensor.node_names)
    perm = g2tensor.permutation.tolist()
    node_names_s = [orig_names_s[p] for p in perm]

    try:
        g_a = pyg_data_to_nx_digraph(g1tensor, original_graphs)
        g_s_original = pyg_data_to_nx_digraph(g2tensor, noise_graphs)
        g_s = g_s_original.copy()
    except ValueError as e:
        print(f"  WARNING: Could not convert graph: {e}")
        return None, None

    if len(g_s.nodes()) == 0:
        print(f"  WARNING: S-graph has no nodes after conversion, skipping visualization")
        return None, None

    if edge_importance.max() > edge_importance.min():
        edge_imp_norm = (edge_importance - edge_importance.min()) / (edge_importance.max() - edge_importance.min())
    else:
        edge_imp_norm = edge_importance

    edges_g_a = list(g_a.edges())
    edge_to_importance_g_a = {}
    for idx, (u, v) in enumerate(edges_g_a):
        if idx < len(edge_imp_norm):
            edge_to_importance_g_a[(u, v)] = edge_imp_norm[idx]
            edge_to_importance_g_a[(v, u)] = edge_imp_norm[idx]

    max_x_a = max(data['center'][0] for _, data in g_a.nodes(data=True))
    min_x_s = min(data['center'][0] for _, data in g_s.nodes(data=True))
    translation_x = (max_x_a - min_x_s) + 10.0
    for _, data in g_s.nodes(data=True):
        data['center'][0] += translation_x
        if 'polygon' in data:
            poly = data['polygon']
            if isinstance(poly, Polygon):
                data['polygon'] = translate(poly, xoff=translation_x)
            else:
                data['polygon'] = Polygon([(x + translation_x, y) for x, y in poly])
        if 'limits' in data:
            data['limits'] = [[x + translation_x, y] for x, y in data['limits']]

    fig, ax = plt.subplots(figsize=(22, 12))
    legend_added = set()

    # A-GRAPH (BIM) - LEFT SIDE
    for (u, v), imp in edge_to_importance_g_a.items():
        if u in g_a.nodes and v in g_a.nodes:
            edge_color = plt.cm.RdYlBu_r(imp)
            linewidth = 1.5 + imp * 4
            ax.plot([g_a.nodes[u]['center'][0], g_a.nodes[v]['center'][0]],
                   [g_a.nodes[u]['center'][1], g_a.nodes[v]['center'][1]],
                   color=edge_color, linewidth=linewidth, alpha=0.8, zorder=2)

    for n, d in g_a.nodes(data=True):
        if d['type'] == 'room' and 'polygon' in d:
            poly = Polygon(d['polygon']) if not isinstance(d['polygon'], Polygon) else d['polygon']
            x, y = poly.exterior.xy
            ax.fill(x, y, alpha=0.15, fc='lightgray', ec='gray', linewidth=0.8, zorder=1)

    for n, d in g_a.nodes(data=True):
        if d['type'] == 'ws':
            ax.scatter(d['center'][0], d['center'][1], color='darkgray', s=60,
                      edgecolors='gray', linewidth=1, zorder=3)
            if 'limits' in d:
                l1, l2 = d['limits']
                ax.plot([l1[0], l2[0]], [l1[1], l2[1]], 'gray', linewidth=1.0, alpha=0.6)

    for n, d in g_a.nodes(data=True):
        if d['type'] == 'room':
            center = d['center']
            ax.scatter(center[0], center[1], color='darkgray', s=80,
                      edgecolors='gray', linewidth=1.5, zorder=3)
            room_label = str(n).split('_')[-2] if '_' in str(n) else str(n)[:10]
            ax.annotate(room_label, (center[0], center[1]),
                       fontsize=7, ha='center', va='center',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # S-GRAPH (Robot) - RIGHT SIDE
    for u, v in g_s.edges():
        ax.plot([g_s.nodes[u]['center'][0], g_s.nodes[v]['center'][0]],
               [g_s.nodes[u]['center'][1], g_s.nodes[v]['center'][1]],
               color='lightgray', linewidth=1.0, alpha=0.5, zorder=1)

    for n, d in g_s.nodes(data=True):
        if d['type'] == 'room' and 'polygon' in d:
            poly = Polygon(d['polygon']) if not isinstance(d['polygon'], Polygon) else d['polygon']
            x, y = poly.exterior.xy
            ax.fill(x, y, alpha=0.15, fc='lightcoral', ec='gray', linewidth=0.8, zorder=1)

    for n, d in g_s.nodes(data=True):
        if d['type'] == 'ws':
            is_target = (n == target_node_id)

            if is_target:
                if 'Target S-node' not in legend_added:
                    ax.scatter(d['center'][0], d['center'][1], c='gold', s=120, marker='o',
                              edgecolors='darkorange', linewidth=2.5, zorder=10,
                              label='Target S-node')
                    legend_added.add('Target S-node')
                else:
                    ax.scatter(d['center'][0], d['center'][1], c='gold', s=120, marker='o',
                              edgecolors='darkorange', linewidth=2.5, zorder=10)
            else:
                if 'S-node' not in legend_added:
                    ax.scatter(d['center'][0], d['center'][1], color='darkgray', s=50,
                              edgecolors='gray', linewidth=1, zorder=3,
                              label='S-node')
                    legend_added.add('S-node')
                else:
                    ax.scatter(d['center'][0], d['center'][1], color='darkgray', s=50,
                              edgecolors='gray', linewidth=1, zorder=3)

            if 'limits' in d:
                l1, l2 = d['limits']
                ax.plot([l1[0], l2[0]], [l1[1], l2[1]], 'gray', linewidth=1.0, alpha=0.6)

    for n, d in g_s.nodes(data=True):
        if d['type'] == 'room':
            is_target = (n == target_node_id)

            if is_target:
                center = d['center']
                if 'Target centroid' not in legend_added:
                    ax.scatter(center[0], center[1], c='gold', s=180, marker='o',
                              edgecolors='darkorange', linewidth=2.5, zorder=10,
                              label='Target centroid')
                    legend_added.add('Target centroid')
                else:
                    ax.scatter(center[0], center[1], c='gold', s=180, marker='o',
                              edgecolors='darkorange', linewidth=2.5, zorder=10)
            else:
                center = d['center']
                if 'Centroid' not in legend_added:
                    ax.scatter(center[0], center[1], color='darkgray', s=80,
                              edgecolors='gray', linewidth=1.5, zorder=3,
                              label='Centroid')
                    legend_added.add('Centroid')
                else:
                    ax.scatter(center[0], center[1], color='darkgray', s=80,
                              edgecolors='gray', linewidth=1.5, zorder=3)

    # LEGEND - No match lines in edge importance
    legend_handles = [
        plt.Rectangle((0,0),1,1, facecolor='lightgray', alpha=0.5, edgecolor='gray'),
        plt.Rectangle((0,0),1,1, facecolor='lightcoral', alpha=0.5, edgecolor='gray'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gold', markersize=10,
                   markeredgecolor='darkorange', linestyle='None'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='darkgray', markersize=8,
                   markeredgecolor='gray', linestyle='None'),
        plt.Line2D([0], [0], color='red', linewidth=3, alpha=0.8),
        plt.Line2D([0], [0], color='blue', linewidth=1.5, alpha=0.8),
        plt.Line2D([0], [0], color='lightgray', linewidth=1.5, alpha=0.6),
    ]
    legend_labels = [
        'A-graph (BIM)',
        'S-graph (Robot)',
        'Target S-node',
        'S-node',
        'High importance edge',
        'Low importance edge',
        'S-graph edge',
    ]
    ax.legend(legend_handles, legend_labels, loc='upper right', fontsize=9, framealpha=0.9, ncol=2)

    sm = ScalarMappable(norm=Normalize(0, 1), cmap=plt.cm.RdYlBu_r)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.5)
    cbar.set_label('EDGE IMPORTANCE (Red=High, Blue=Low)', fontsize=12, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)

    ax.set_title(f"{title}", fontsize=14, fontweight='bold')
    ax.set_xlabel('X (meters)', fontsize=12)
    ax.set_ylabel('Y (meters)', fontsize=12)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.tight_layout()
    return fig, ax