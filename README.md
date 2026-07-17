# Explainable Graph Neural Networks for Graph Matching in Building Information Modelling

## Project Topic
Experiments with Graph Neural Networks in the Construction Industry (Building Information Modelling)

## Project Supervisors
- Professor Marco Cococcioni
- Matteo Giorgi

## Group Members
- Reza Almassi  
- Ronald Omoding  
- Tewodros Abere Muche  

## Project Description

This project aims to improve the performance and interpretability of a graph‑matching model for hierarchical scene graphs (rooms + wall surfaces).  
We evaluate multiple GNN architectures, compare their performance on a preprocessed MSD dataset, and apply explainability techniques to understand which nodes and edges contribute most to incorrect predictions.

### Inspired by:
- Ndulue et al. (2026) — *Learning-Based Hierarchical Scene Graph Matching for Robot Localization Leveraging Prior Maps*  
- Shaheer et al. (2023) — *Graph-based Global Robot Localization Informing Situational Graphs with Architectural Graphs*

### Core Objectives
- Benchmark multiple GNN architectures for graph matching
    - **GATv2** (Brody et al., 2021)  
    - **GCN** (Kipf & Welling, 2017)  
    - **GraphSAGE** (Hamilton et al., 2017)  
    - **GIN** (Xu et al., 2019)  
    - **Graph Transformer** (Shi et al., 2020)
    - **GINE** (Hu et al., 2020)
    - **RGCN** (Schlichtkrull et al et al., 2017)
- Evaluate performance on preprocessed version of the MSD dataset  
- Apply GNNExplainer to interprete predictions  
- Diagnose incorrect matches and highlight model weaknesses  

## Dataset
The dataset consists of paired graphs from:

1. A-graphs (BIM): Building Information Models with room polygons and wall segments

2. S-graphs (Robot): Robot perception data with semantic features

### Dataset Format
Each sample is a tuple (Data1, Data2, PermutationMatrix):

- Data1: PyG Data object for A-graph (BIM)

- Data2: PyG Data object for S-graph (Robot)

- PermutationMatrix: Ground truth node correspondence (N_A × N_S)

### Node Features
7-dimensional feature vector per node:

- **Type_Room**: [1, 0] for room nodes

- **Type_WS**: [0, 1] for wall segment nodes

- **Centroid_X**: X coordinate

- **Centroid_Y**: Y coordinate

- **Normal_X**: Outward-facing normal X

- **Normal_Y**: Outward-facing normal Y

- **Segment_Length**: Wall segment length (-1 for rooms)

### Edge Features
The graph contains three distinct edge types:
- **Room-WS**: Connects room to wall segment
- **Room-Room**: Connects two rooms
- **WS-WS**: Connects two wall segments

## Training
```bash
python train.py --model gatv2 --data_path /msd_data --checkpoint_dir ./checkpoints
```
## Evaluation
```bash
python evaluate.py --model gatv2 --data_path /msd_data --checkpoint_dir ./checkpoints --visualize
```

## Model Explainability
The repository includes comprehensive explainability features:

### GNNExplainer Integration
- Node Importance: Identifies which nodes in the BIM graph are most influential for a prediction

- Edge Importance: Identifies which graph edges are most important

- Feature Importance: Shows which node features contribute to the decision

### Visualization Types
- Node Importance Plots: Color-coded nodes showing importance (red = high, blue = low)

- Edge Importance Plots: Thick/thin edges indicating importance

- Matching Visualization: Side-by-side graphs with matching lines

```bash
python scripts/explain_model.py \
    --model gine \
    --data_path /msd_data \
    --checkpoint_dir ./checkpoints \
    --output_dir ./explanations \
    --num_samples 5 \
    --explain_wrong \
    --explain_correct \
    --visualize
```
Explanation Options:
```bash
 --model: Model architecture to explain
 --data_path: Path to the MSD dataset
 --checkpoint_dir: Directory containing model checkpoints
 --output_dir: Directory for explanation outputs
 --num_samples: Number of samples to analyze
 --epochs: Number of epochs for GNNExplainer
 --explain_wrong: Explain misclassified nodes
 --explain_correct: Explain correctly classified nodes
 --visualize: Generate visualization plots
```
