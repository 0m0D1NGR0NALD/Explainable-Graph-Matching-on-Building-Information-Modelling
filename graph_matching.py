GNN_PATH = './GNN/'
import os
from dataset_utility import *
if not os.path.exists(GNN_PATH):
    os.makedirs(GNN_PATH)

# Construct the folders if they don't exist
create_dir_structure(GNN_PATH)

# Graph Matching dataset creation

# A-graphs
gm_path = os.path.join(GNN_PATH, "raw", "graph_matching")
original_graphs, _, _ = deserialize_MSD_dataset(data_path=gm_path, original_path="equal")
# S-graphs with incremental noise
gm_path = os.path.join(GNN_PATH, "raw", "partial_graph_matching")
noise_graphs_15, _, _ = deserialize_MSD_dataset(data_path=gm_path, original_path="ws_room_dropout_noise_inc/15")
noise_graphs_35, _, _ = deserialize_MSD_dataset(data_path=gm_path, original_path="ws_room_dropout_noise_inc/35")
noise_graphs_55, _, _ = deserialize_MSD_dataset(data_path=gm_path, original_path="ws_room_dropout_noise_inc/55")
noise_graphs_75, _, _ = deserialize_MSD_dataset(data_path=gm_path, original_path="ws_room_dropout_noise_inc/75")
noise_graphs_95, _, _ = deserialize_MSD_dataset(data_path=gm_path, original_path="ws_room_dropout_noise_inc/95")

# Check the number of graphs
assert len(noise_graphs_15) == len(noise_graphs_35) == len(noise_graphs_55) == len(noise_graphs_75) == len(noise_graphs_95), "All noise graphs should have the same number of graphs"
noise_graphs = noise_graphs_15 + noise_graphs_35 + noise_graphs_55 + noise_graphs_75 + noise_graphs_95

print(f"Number of original graphs: {len(noise_graphs)}")

plot_a_graph([noise_graphs_15[0]],path=os.path.join(gm_path, "ws_room_dropout_noise_inc", "apartment_15.png"), viz_rooms=True, viz_ws=True, viz_openings=False, viz_room_connection=True, viz_normals=False, viz_room_normals=True, viz_walls=True)
plot_a_graph([noise_graphs_35[0]],path=os.path.join(gm_path, "ws_room_dropout_noise_inc", "apartment_35.png"), viz_rooms=True, viz_ws=True, viz_openings=False, viz_room_connection=True, viz_normals=False, viz_room_normals=True, viz_walls=True)
plot_a_graph([noise_graphs_55[0]],path=os.path.join(gm_path, "ws_room_dropout_noise_inc", "apartment_55.png"), viz_rooms=True, viz_ws=True, viz_openings=False, viz_room_connection=True, viz_normals=False, viz_room_normals=True, viz_walls=True)
plot_a_graph([noise_graphs_75[0]],path=os.path.join(gm_path, "ws_room_dropout_noise_inc", "apartment_75.png"), viz_rooms=True, viz_ws=True, viz_openings=False, viz_room_connection=True, viz_normals=False, viz_room_normals=True, viz_walls=True)
plot_a_graph([noise_graphs_95[0]],path=os.path.join(gm_path, "ws_room_dropout_noise_inc", "apartment_95.png"), viz_rooms=True, viz_ws=True, viz_openings=False, viz_room_connection=True, viz_normals=False, viz_room_normals=True, viz_walls=True)

# Generate G1,G2,GT dataset
pair_gt_list = []

for i, g2 in enumerate(tqdm(noise_graphs, desc="Pair graph generation", unit="graph", ncols=80)):
    generate_matching_pair_as_data(original_graphs[i % len(original_graphs)], g2, pair_gt_list)

assert len(pair_gt_list) == len(noise_graphs_15) * 5, "Pair GT list should contain 5 times the number of noise graphs"

train, val, test = split_graphs_stratified(pair_gt_list)

describe(train, "TRAIN")
describe(val,   "VAL")
describe(test,  "TEST")

# compute mean and std
mean, std = compute_mean_std(train)

# Normalizzazione dei set
train_pairs_norm = normalize_data_pairs(train, mean, std)
val_pairs_norm = normalize_data_pairs(val, mean, std)
test_pairs_norm = normalize_data_pairs(test, mean, std)

gm_equal_preprocessed_path = os.path.join(GNN_PATH, "preprocessed", "partial_graph_matching", "ws_room_dropout_noise_inc")
serialize_graph_matching_dataset(
    train_pairs_norm,
    gm_equal_preprocessed_path,
    "train_dataset.pkl"
)
serialize_graph_matching_dataset(
    val_pairs_norm,
    gm_equal_preprocessed_path,
    "valid_dataset.pkl"
)
serialize_graph_matching_dataset(
    test_pairs_norm,
    gm_equal_preprocessed_path,
    "test_dataset.pkl"
)
serialize_graph_matching_dataset(
    noise_graphs,
    gm_equal_preprocessed_path,
    "noise.pkl"
)

g1_out, g2_perm, gt_perm = train[0]

print(g1_out)
print("G1 nodes:", g1_out.x[0])
print(g2_perm)
print("G2 permuted nodes:", g2_perm.x[0])
print("Ground truth permutation:\n", gt_perm[0])

# Visualize the two graphs
plot_two_graphs_with_matching(
    [g1_out, g2_perm],
    gt_perm=gt_perm,
    pred_perm=gt_perm,
    original_graphs=original_graphs,
    noise_graphs=noise_graphs,
    viz_rooms=True,
    viz_ws=True,
    match_display="all",
    path=os.path.join(gm_equal_preprocessed_path, "gt.png")
)

# Load preprocessed dataset
gm_equal_preprocessed_path = os.path.join(GNN_PATH, "preprocessed", "graph_matching", "equal")
gm_local_preprocessed_path = os.path.join(GNN_PATH, "preprocessed", "partial_graph_matching", "ws_room_dropout_noise_inc")
models_path = os.path.join(GNN_PATH, 'models', "partial_graph_matching", "ws_room_dropout_noise_inc")

original_graphs = deserialize_graph_matching_dataset(
    gm_equal_preprocessed_path,
    "original.pkl"
)
noise_graphs = deserialize_graph_matching_dataset(
    gm_local_preprocessed_path,
    "noise.pkl"
)

train_list = deserialize_graph_matching_dataset(
    gm_local_preprocessed_path,
    "train_dataset.pkl"
)
val_list = deserialize_graph_matching_dataset(
    gm_local_preprocessed_path,
    "valid_dataset.pkl"
)
test_list = deserialize_graph_matching_dataset(
    gm_local_preprocessed_path,
    "test_dataset.pkl"
)

d1,d2,gt = train_list[0]
print(d1)
print(d2)
print(gt.shape)

# Visualize the two graphs
plot_two_graphs_with_matching(
    [d1,d2],
    gt_perm=gt,
    pred_perm=gt,
    original_graphs=original_graphs,
    noise_graphs=noise_graphs,
    viz_rooms=True,
    viz_ws=True,
    match_display="all",
)

# plot d2 graph with plot_a_graph
g2 = pyg_data_to_nx_digraph(d2, original_graphs)
plot_a_graph([g2], viz_rooms=True, viz_ws=True, viz_openings=False, viz_room_connection=True, viz_normals=False, viz_room_normals=True, viz_walls=True)

train_dataset = GraphMatchingDataset(train_list)
val_dataset = GraphMatchingDataset(val_list)
test_dataset = GraphMatchingDataset(test_list)

batch_size = 16
# Loader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_pyg_matching, generator=g)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_pyg_matching)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_pyg_matching)
