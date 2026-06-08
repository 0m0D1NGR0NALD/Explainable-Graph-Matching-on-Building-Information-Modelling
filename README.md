# Explainable Graph Neural Networks for Graph Matching in Building Information Modelling

## Project Topic
Experiments with Graph Neural Networks in the Construction Industry (Building Information Modelling)

## Group Members
- Reza Almassi  
- Ronald Omoding  
- Tewodros Abere Muche  

---

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
- Evaluate performance on the MSD dataset  
- Apply GNNExplainer to interpret predictions  
- Diagnose incorrect matches and model weaknesses  
