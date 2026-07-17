import copy
import matplotlib.pyplot as plt
import networkx as nx
from shapely.geometry import Polygon
from shapely.affinity import translate
from torch_geometric.data import Batch

from .dataset import pyg_data_to_nx_digraph


def plot_a_graph(graphs_list, ax=None, viz_rooms=True, viz_ws=True,
                 viz_room_connection=True, viz_normals=False,
                 viz_room_normals=False, viz_walls=True, title=None):
    """
    Visualizes geometries, wall segments, and graph edges for multiple apartments in 2D.
    """
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(12, 10))

    legend_added = set()
    normal_added = False

    for graphs in graphs_list:
        # Visualize room polygons
        if viz_rooms:
            room_nodes = [n for n, d in graphs.nodes(data=True) if d['type'] == 'room']
            for idx, room_node in enumerate(room_nodes):
                room_data = graphs.nodes[room_node]
                if 'polygon' in room_data:
                    room_polygon = Polygon(room_data['polygon'])
                    x, y = room_polygon.exterior.xy
                    if "Room polygon" not in legend_added:
                        ax.plot(x, y, color='black', alpha=0.3, linewidth=2, label='Room polygon')
                        legend_added.add("Room polygon")
                    else:
                        ax.plot(x, y, color='black', alpha=0.3, linewidth=2)
                if "Room centroid" not in legend_added:
                    ax.scatter(room_data['center'][0], room_data['center'][1], color='blue', s=100,
                              edgecolors='darkblue', linewidth=2, zorder=3, label='Room centroid')
                    legend_added.add("Room centroid")
                else:
                    ax.scatter(room_data['center'][0], room_data['center'][1], color='blue', s=100,
                              edgecolors='darkblue', linewidth=2, zorder=3)

        # Visualize WS nodes
        if viz_ws:
            ws_nodes = [n for n, d in graphs.nodes(data=True) if d['type'] == 'ws']
            for idx, wn in enumerate(ws_nodes):
                ws_data = graphs.nodes[wn]
                if "WS" not in legend_added:
                    ax.scatter(ws_data['center'][0], ws_data['center'][1], color='red', s=40,
                              edgecolors='darkred', linewidth=1.5, zorder=3, label='Wall segment')
                    legend_added.add("WS")
                else:
                    ax.scatter(ws_data['center'][0], ws_data['center'][1], color='red', s=40,
                              edgecolors='darkred', linewidth=1.5, zorder=3)

                if viz_room_normals:
                    if not normal_added:
                        ax.arrow(ws_data['center'][0], ws_data['center'][1],
                                ws_data['normal'][0], ws_data['normal'][1],
                                head_width=0.15, head_length=0.15, fc='green', ec='green',
                                alpha=0.7, label='Normal')
                        normal_added = True
                    else:
                        ax.arrow(ws_data['center'][0], ws_data['center'][1],
                                ws_data['normal'][0], ws_data['normal'][1],
                                head_width=0.15, head_length=0.15, fc='green', ec='green', alpha=0.7)

                if 'limits' in ws_data and viz_walls:
                    limit_1, limit_2 = ws_data['limits']
                    ax.plot([limit_1[0], limit_2[0]], [limit_1[1], limit_2[1]],
                            color='black', linewidth=2.0, alpha=0.8)

            # Draw WS edges
            if viz_ws:
                ws_edges = [(u, v) for u, v, d in graphs.edges(data=True)
                           if d.get('type') in ['ws_same_room', 'ws_belongs_room']]
                for idx, edge in enumerate(ws_edges):
                    start_node = graphs.nodes[edge[0]]
                    end_node = graphs.nodes[edge[1]]
                    if "WS edge" not in legend_added:
                        ax.plot([start_node['center'][0], end_node['center'][0]],
                               [start_node['center'][1], end_node['center'][1]],
                               color='gray', linestyle='--', alpha=0.6, linewidth=1.5, label='WS connection')
                        legend_added.add("WS edge")
                    else:
                        ax.plot([start_node['center'][0], end_node['center'][0]],
                               [start_node['center'][1], end_node['center'][1]],
                               color='gray', linestyle='--', alpha=0.6, linewidth=1.5)

        # Visualize room connections
        if viz_room_connection:
            connection_edges = [(u, v) for u, v, d in graphs.edges(data=True)
                               if d.get('type') == 'connected']
            for idx, edge in enumerate(connection_edges):
                start_node = graphs.nodes[edge[0]]
                end_node = graphs.nodes[edge[1]]
                if "Room connection" not in legend_added:
                    ax.plot([start_node['center'][0], end_node['center'][0]],
                           [start_node['center'][1], end_node['center'][1]],
                           color='blue', linestyle='-', alpha=0.5, linewidth=1, label='Room adjacency')
                    legend_added.add("Room connection")
                else:
                    ax.plot([start_node['center'][0], end_node['center'][0]],
                           [start_node['center'][1], end_node['center'][1]],
                           color='blue', linestyle='-', alpha=0.5, linewidth=1)

    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('X (meters)', fontsize=12)
    ax.set_ylabel('Y (meters)', fontsize=12)
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    return ax


def plot_two_graphs_with_matching(graphs_list, gt_perm, original_graphs,
                                   pred_perm=None, noise_graphs=None,
                                   viz_rooms=True, viz_ws=True,
                                   viz_room_connection=True,
                                   viz_normals=False, viz_room_normals=False,
                                   match_display="all", title=None, save_path=None):
    """
    Visualizes two graphs side-by-side with matching lines.
    Green lines = correct matches, Red lines = wrong matches.
    """
    assert match_display in {"all", "correct", "wrong"}, "match_display must be 'all', 'correct', or 'wrong'"
    assert len(graphs_list) == 2, "graphs_list must contain exactly two graphs."

    if noise_graphs is None:
        noise_graphs = original_graphs

    g1tensor, g2tensor = copy.deepcopy(graphs_list[0]), copy.deepcopy(graphs_list[1])
    node_names_a = list(g1tensor.node_names)
    orig_names_s = list(g2tensor.node_names)
    perm = g2tensor.permutation.tolist()
    node_names_s = [orig_names_s[p] for p in perm]

    g_a = pyg_data_to_nx_digraph(g1tensor, original_graphs)
    g_s_original = pyg_data_to_nx_digraph(g2tensor, noise_graphs)
    g_s = g_s_original.copy()

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

    fig, ax = plt.subplots(figsize=(18, 10))

    matched_s_nodes = set()
    if pred_perm is not None:
        if pred_perm.shape[0] == len(node_names_a) and pred_perm.shape[1] == len(node_names_s):
            pred_perm = pred_perm.T
        for s_idx in range(min(pred_perm.shape[0], len(node_names_s))):
            row = pred_perm[s_idx]
            if row.sum().item() > 0:
                matched_s_nodes.add(node_names_s[s_idx])

    if gt_perm.shape[0] == len(node_names_a) and gt_perm.shape[1] == len(node_names_s):
        gt_perm = gt_perm.T

    # Plot A-graph (BIM) - LEFT SIDE
    color_room_a = 'lightblue'
    color_ws_a = 'red'
    prefix_a = "A-graph (BIM)"

    if viz_rooms:
        room_label_added = False
        centroid_label_added = False

        for n, d in g_a.nodes(data=True):
            if d['type'] == 'room' and 'polygon' in d:
                poly = Polygon(d['polygon']) if not isinstance(d['polygon'], Polygon) else d['polygon']
                x, y = poly.exterior.xy

                if not room_label_added:
                    ax.fill(x, y, color=color_room_a, alpha=0.3, label=f"{prefix_a} room")
                    room_label_added = True
                else:
                    ax.fill(x, y, color=color_room_a, alpha=0.3)

                if not centroid_label_added:
                    ax.scatter(d['center'][0], d['center'][1], color='blue', s=120,
                              edgecolors='darkblue', linewidth=2, zorder=3,
                              label=f"{prefix_a} centroid")
                    centroid_label_added = True
                else:
                    ax.scatter(d['center'][0], d['center'][1], color='blue', s=120,
                              edgecolors='darkblue', linewidth=2, zorder=3)

    if viz_ws:
        ws_label_added = False
        wall_label_added = False

        for n, d in g_a.nodes(data=True):
            if d['type'] == 'ws':
                if not ws_label_added:
                    ax.scatter(d['center'][0], d['center'][1], color=color_ws_a, s=50,
                              edgecolors='darkred', linewidth=1.5, zorder=3,
                              label=f"{prefix_a} WS")
                    ws_label_added = True
                else:
                    ax.scatter(d['center'][0], d['center'][1], color=color_ws_a, s=50,
                              edgecolors='darkred', linewidth=1.5, zorder=3)

                if 'limits' in d:
                    limit1, limit2 = d['limits']
                    if not wall_label_added:
                        ax.plot([limit1[0], limit2[0]], [limit1[1], limit2[1]],
                               color='black', linewidth=2.0, alpha=0.8,
                               label=f"{prefix_a} wall")
                        wall_label_added = True
                    else:
                        ax.plot([limit1[0], limit2[0]], [limit1[1], limit2[1]],
                               color='black', linewidth=2.0, alpha=0.8)

    if viz_room_connection:
        conn_label_added = False
        connection_edges = [(u, v) for u, v, d in g_a.edges(data=True)
                           if d.get('type') == 'connected']
        for u, v in connection_edges:
            start_node = g_a.nodes[u]
            end_node = g_a.nodes[v]
            if not conn_label_added:
                ax.plot([start_node['center'][0], end_node['center'][0]],
                       [start_node['center'][1], end_node['center'][1]],
                       color='blue', linestyle='-', alpha=0.5, linewidth=1,
                       label='Room adjacency')
                conn_label_added = True
            else:
                ax.plot([start_node['center'][0], end_node['center'][0]],
                       [start_node['center'][1], end_node['center'][1]],
                       color='blue', linestyle='-', alpha=0.5, linewidth=1)

    # Plot S-graph (Robot) - RIGHT SIDE
    color_room_s = 'navajowhite'
    color_ws_s = 'purple'
    prefix_s = "S-graph (Robot)"

    if viz_rooms:
        room_label_added = False
        centroid_matched_label_added = False
        centroid_unmatched_label_added = False

        for n, d in g_s.nodes(data=True):
            if d['type'] == 'room' and 'polygon' in d:
                poly = Polygon(d['polygon']) if not isinstance(d['polygon'], Polygon) else d['polygon']
                x, y = poly.exterior.xy

                if not room_label_added:
                    ax.fill(x, y, color=color_room_s, alpha=0.3, label=f"{prefix_s} room")
                    room_label_added = True
                else:
                    ax.fill(x, y, color=color_room_s, alpha=0.3)

                is_matched = n in matched_s_nodes

                if not is_matched:
                    if not centroid_unmatched_label_added:
                        ax.scatter(d['center'][0], d['center'][1],
                                  color='none', edgecolor='red', s=140,
                                  linewidth=2.5, alpha=0.9, zorder=2)
                        ax.scatter(d['center'][0], d['center'][1],
                                  color='dimgray', s=120,
                                  edgecolors='black', linewidth=2, zorder=3,
                                  label=f"{prefix_s} centroid (unmatched)")
                        centroid_unmatched_label_added = True
                    else:
                        ax.scatter(d['center'][0], d['center'][1],
                                  color='none', edgecolor='red', s=140,
                                  linewidth=2.5, alpha=0.9, zorder=2)
                        ax.scatter(d['center'][0], d['center'][1],
                                  color='dimgray', s=120,
                                  edgecolors='black', linewidth=2, zorder=3)
                else:
                    if not centroid_matched_label_added:
                        ax.scatter(d['center'][0], d['center'][1], color='#40E0D0', s=120,
                                  edgecolors='#008080', linewidth=2, zorder=3,
                                  label=f"{prefix_s} centroid")
                        centroid_matched_label_added = True
                    else:
                        ax.scatter(d['center'][0], d['center'][1], color='#40E0D0', s=120,
                                  edgecolors='#008080', linewidth=2, zorder=3)

    if viz_ws:
        ws_matched_label_added = False
        ws_unmatched_label_added = False
        wall_label_added = False

        for n, d in g_s.nodes(data=True):
            if d['type'] == 'ws':
                is_matched = n in matched_s_nodes

                if not is_matched:
                    if not ws_unmatched_label_added:
                        ax.scatter(d['center'][0], d['center'][1],
                                  color='none', edgecolor='red', s=80,
                                  linewidth=2.5, alpha=0.9, zorder=2)
                        ax.scatter(d['center'][0], d['center'][1],
                                  color='dimgray', s=50,
                                  edgecolors='black', linewidth=1.5, zorder=3,
                                  label=f"{prefix_s} WS (unmatched)")
                        ws_unmatched_label_added = True
                    else:
                        ax.scatter(d['center'][0], d['center'][1],
                                  color='none', edgecolor='red', s=80,
                                  linewidth=2.5, alpha=0.9, zorder=2)
                        ax.scatter(d['center'][0], d['center'][1],
                                  color='dimgray', s=50,
                                  edgecolors='black', linewidth=1.5, zorder=3)
                else:
                    if not ws_matched_label_added:
                        ax.scatter(d['center'][0], d['center'][1], color=color_ws_s, s=50,
                                  edgecolors='purple', linewidth=1.5, zorder=3,
                                  label=f"{prefix_s} WS")
                        ws_matched_label_added = True
                    else:
                        ax.scatter(d['center'][0], d['center'][1], color=color_ws_s, s=50,
                                  edgecolors='purple', linewidth=1.5, zorder=3)

                if 'limits' in d:
                    limit1, limit2 = d['limits']
                    if not wall_label_added:
                        ax.plot([limit1[0], limit2[0]], [limit1[1], limit2[1]],
                               color='black', linewidth=2.0, alpha=0.8,
                               label=f"{prefix_s} wall")
                        wall_label_added = True
                    else:
                        ax.plot([limit1[0], limit2[0]], [limit1[1], limit2[1]],
                               color='black', linewidth=2.0, alpha=0.8)

    if viz_ws:
        ws_edges = [(u, v) for u, v, d in g_s.edges(data=True)
                   if d.get('type') in ['ws_same_room', 'ws_belongs_room']]
        edge_label_added = False
        for u, v in ws_edges:
            start_node = g_s.nodes[u]
            end_node = g_s.nodes[v]
            if not edge_label_added:
                ax.plot([start_node['center'][0], end_node['center'][0]],
                       [start_node['center'][1], end_node['center'][1]],
                       color='gray', linestyle='--', alpha=0.6, linewidth=1.5,
                       label=f"{prefix_s} WS connection")
                edge_label_added = True
            else:
                ax.plot([start_node['center'][0], end_node['center'][0]],
                       [start_node['center'][1], end_node['center'][1]],
                       color='gray', linestyle='--', alpha=0.6, linewidth=1.5)

    # MATCHING LINES (S → A)
    if pred_perm is not None:
        correct_label_added = False
        wrong_label_added = False

        for s_idx in range(pred_perm.shape[0]):
            if s_idx >= gt_perm.shape[0]:
                continue
            if gt_perm[s_idx].sum().item() == 0:
                continue

            row = pred_perm[s_idx]
            if row.sum().item() == 0:
                continue

            a_idx = row.argmax().item()
            s_node_name = node_names_s[s_idx]
            if s_node_name not in g_s.nodes:
                continue
            if a_idx >= len(node_names_a):
                continue
            a_node_name = node_names_a[a_idx]
            if a_node_name not in g_a.nodes:
                continue

            pt_s = g_s.nodes[s_node_name]['center']
            pt_a = g_a.nodes[a_node_name]['center']

            is_correct = (a_idx < gt_perm.shape[1] and gt_perm[s_idx, a_idx] == 1)

            if match_display == "correct" and not is_correct:
                continue
            if match_display == "wrong" and is_correct:
                continue

            color = 'green' if is_correct else 'red'

            if color == 'green' and not correct_label_added:
                ax.plot([pt_s[0], pt_a[0]], [pt_s[1], pt_a[1]],
                       color=color, linestyle='-', alpha=0.7, linewidth=2.5,
                       label='Correct match')
                correct_label_added = True
            elif color == 'green':
                ax.plot([pt_s[0], pt_a[0]], [pt_s[1], pt_a[1]],
                       color=color, linestyle='-', alpha=0.7, linewidth=2.5)
            elif color == 'red' and not wrong_label_added:
                ax.plot([pt_s[0], pt_a[0]], [pt_s[1], pt_a[1]],
                       color=color, linestyle='-', alpha=0.7, linewidth=2.5,
                       label='Wrong match')
                wrong_label_added = True
            else:
                ax.plot([pt_s[0], pt_a[0]], [pt_s[1], pt_a[1]],
                       color=color, linestyle='-', alpha=0.7, linewidth=2.5)

    ax.set_title(title if title else "Graph Matching Results (S → A)", fontsize=16, fontweight='bold')
    ax.set_xlabel('X (meters)', fontsize=12)
    ax.set_ylabel('Y (meters)', fontsize=12)
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9, ncol=2)
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")

    plt.tight_layout()
    return fig, ax


def explore_edge_data(dataset, name="Dataset"):
    """
    Analyze edge types in the graph matching dataset.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    print(f"EDGE TYPE ANALYSIS: {name}")

    a_room_ws_counts = []
    a_room_room_counts = []
    a_ws_ws_counts = []
    s_room_ws_counts = []
    s_room_room_counts = []
    s_ws_ws_counts = []

    for idx, (g1, g2, P_gt) in enumerate(dataset):
        room_ws = 0
        room_room = 0
        ws_ws = 0

        node_names = g1.node_names
        edge_index = g1.edge_index.t().tolist()

        for u, v in edge_index:
            u_name = node_names[u]
            v_name = node_names[v]

            u_is_room = 'centroid' in u_name or ('room' in u_name.lower() and 'ws' not in u_name)
            v_is_room = 'centroid' in v_name or ('room' in v_name.lower() and 'ws' not in v_name)

            if u_is_room != v_is_room:
                room_ws += 1
            elif u_is_room and v_is_room:
                room_room += 1
            else:
                ws_ws += 1

        a_room_ws_counts.append(room_ws)
        a_room_room_counts.append(room_room)
        a_ws_ws_counts.append(ws_ws)

        room_ws = 0
        room_room = 0
        ws_ws = 0

        node_names = g2.node_names
        edge_index = g2.edge_index.t().tolist()

        for u, v in edge_index:
            u_name = node_names[u]
            v_name = node_names[v]

            u_is_room = 'centroid' in u_name or ('room' in u_name.lower() and 'ws' not in u_name)
            v_is_room = 'centroid' in v_name or ('room' in v_name.lower() and 'ws' not in v_name)

            if u_is_room != v_is_room:
                room_ws += 1
            elif u_is_room and v_is_room:
                room_room += 1
            else:
                ws_ws += 1

        s_room_ws_counts.append(room_ws)
        s_room_room_counts.append(room_room)
        s_ws_ws_counts.append(ws_ws)

    print(f"\n{'Graph Type':<15} {'Room-WS':<12} {'Room-Room':<12} {'WS-WS':<12}")
    print(f"{'A-graph (avg)':<15} {np.mean(a_room_ws_counts):<12.1f} {np.mean(a_room_room_counts):<12.1f} {np.mean(a_ws_ws_counts):<12.1f}")
    print(f"{'S-graph (avg)':<15} {np.mean(s_room_ws_counts):<12.1f} {np.mean(s_room_room_counts):<12.1f} {np.mean(s_ws_ws_counts):<12.1f}")

    total_room_ws = sum(a_room_ws_counts) + sum(s_room_ws_counts)
    total_room_room = sum(a_room_room_counts) + sum(s_room_room_counts)
    total_ws_ws = sum(a_ws_ws_counts) + sum(s_ws_ws_counts)
    total_edges = total_room_ws + total_room_room + total_ws_ws

    pct_room_ws = total_room_ws / total_edges * 100
    pct_room_room = total_room_room / total_edges * 100
    pct_ws_ws = total_ws_ws / total_edges * 100

    print(f"\nEdge Type Percentages (overall):")
    print(f"  Room-WS: {pct_room_ws:.1f}%")
    print(f"  Room-Room: {pct_room_room:.1f}%")
    print(f"  WS-WS: {pct_ws_ws:.1f}%")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    axes[0, 0].hist(a_room_ws_counts, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Count')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('A-graph: Room-WS Edges')
    axes[0, 0].axvline(np.mean(a_room_ws_counts), color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {np.mean(a_room_ws_counts):.1f}')
    axes[0, 0].axvline(np.median(a_room_ws_counts), color='green', linestyle='--', linewidth=2,
                       label=f'Median: {np.median(a_room_ws_counts):.1f}')
    axes[0, 0].legend()

    axes[0, 1].hist(a_room_room_counts, bins=20, color='coral', edgecolor='black', alpha=0.7)
    axes[0, 1].set_xlabel('Count')
    axes[0, 1].set_title('A-graph: Room-Room Edges')
    axes[0, 1].axvline(np.mean(a_room_room_counts), color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {np.mean(a_room_room_counts):.1f}')
    axes[0, 1].axvline(np.median(a_room_room_counts), color='green', linestyle='--', linewidth=2,
                       label=f'Median: {np.median(a_room_room_counts):.1f}')
    axes[0, 1].legend()

    axes[0, 2].hist(a_ws_ws_counts, bins=20, color='green', edgecolor='black', alpha=0.7)
    axes[0, 2].set_xlabel('Count')
    axes[0, 2].set_title('A-graph: WS-WS Edges')
    axes[0, 2].axvline(np.mean(a_ws_ws_counts), color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {np.mean(a_ws_ws_counts):.1f}')
    axes[0, 2].axvline(np.median(a_ws_ws_counts), color='green', linestyle='--', linewidth=2,
                       label=f'Median: {np.median(a_ws_ws_counts):.1f}')
    axes[0, 2].legend()

    axes[1, 0].hist(s_room_ws_counts, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
    axes[1, 0].set_xlabel('Count')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('S-graph: Room-WS Edges')
    axes[1, 0].axvline(np.mean(s_room_ws_counts), color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {np.mean(s_room_ws_counts):.1f}')
    axes[1, 0].axvline(np.median(s_room_ws_counts), color='green', linestyle='--', linewidth=2,
                       label=f'Median: {np.median(s_room_ws_counts):.1f}')
    axes[1, 0].legend()

    axes[1, 1].hist(s_room_room_counts, bins=20, color='coral', edgecolor='black', alpha=0.7)
    axes[1, 1].set_xlabel('Count')
    axes[1, 1].set_title('S-graph: Room-Room Edges')
    axes[1, 1].axvline(np.mean(s_room_room_counts), color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {np.mean(s_room_room_counts):.1f}')
    axes[1, 1].axvline(np.median(s_room_room_counts), color='green', linestyle='--', linewidth=2,
                       label=f'Median: {np.median(s_room_room_counts):.1f}')
    axes[1, 1].legend()

    axes[1, 2].hist(s_ws_ws_counts, bins=20, color='green', edgecolor='black', alpha=0.7)
    axes[1, 2].set_xlabel('Count')
    axes[1, 2].set_title('S-graph: WS-WS Edges')
    axes[1, 2].axvline(np.mean(s_ws_ws_counts), color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {np.mean(s_ws_ws_counts):.1f}')
    axes[1, 2].axvline(np.median(s_ws_ws_counts), color='green', linestyle='--', linewidth=2,
                       label=f'Median: {np.median(s_ws_ws_counts):.1f}')
    axes[1, 2].legend()

    plt.suptitle(f'{name} - Edge Type Distribution (A-graph vs S-graph)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    return {
        'a_room_ws_avg': np.mean(a_room_ws_counts),
        'a_room_room_avg': np.mean(a_room_room_counts),
        'a_ws_ws_avg': np.mean(a_ws_ws_counts),
        's_room_ws_avg': np.mean(s_room_ws_counts),
        's_room_room_avg': np.mean(s_room_room_counts),
        's_ws_ws_avg': np.mean(s_ws_ws_counts),
        'pct_room_ws': pct_room_ws,
        'pct_room_room': pct_room_room,
        'pct_ws_ws': pct_ws_ws
    }