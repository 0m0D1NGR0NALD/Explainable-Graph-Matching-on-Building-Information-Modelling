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
- Evaluate performance on the MSD dataset  
- Apply GNNExplainer to interpret predictions  
- Diagnose incorrect matches and model weaknesses  

---

# Project Structure

### 1. Setup & Utilities
- Environment initialization  
- Helper functions for batching, visualization, and evaluation  

### 2. Dataset Loading & Assessment
- Load preprocessed MSD dataset  
- Inspect graph sizes, node types, and feature distributions  

### 3. GNN Architectures Implemented
- **GATv2** (Brody et al., 2021)  
- **GCN** (Kipf & Welling, 2017)  
- **GraphSAGE** (Hamilton et al., 2017)  
- **GIN** (Xu et al., 2019)  
- **Graph Transformer** (Shi et al., 2020)

A unified comparison module evaluates all architectures on identical data splits.

### 4. Explainability with GNNExplainer
GNNExplainer is a model‑agnostic method that explains the predictions of any Graph Neural Network (GNN).  
We use it to identify:
- Important A‑graph nodes  
- Critical edges  
- Feature dimensions influencing predictions  

### 5. Hyperparameter Optimization (Optuna)
- Automated search over learning rate, hidden dimensions, dropout, and layers  
- Best model selection and evaluation  

---

# GNNExplainer: Mathematical Foundations

### 1. Mutual Information Objective


\[
\max_{G_S} MI(Y, (G_S, X_S)) = H(Y) - H(Y \mid G_S, X_S)
\]



### 2. Conditional Entropy


\[
H(Y \mid G_S, X_S) = -\mathbb{E}[\log P_\Phi(Y \mid G_S, X_S)]
\]



### 3. Variational Mask Approximation


\[
\mathbb{E}[G_S] = A_c \odot \sigma(M)
\]



### 4. Label‑Specific Cross‑Entropy Objective


\[
\min_M -\sum_{c=1}^C \mathbf{1}[y=c] \log P_\Phi(Y=y \mid A_c \odot \sigma(M), X_c)
\]



---

# Node Classification for Graph Matching

We freeze the pretrained GNN encoder and train a classifier head:



\[
\text{logits} = \text{NodeClassifier}(\text{GNN}(x, S))
\]



Labels are derived from the ground‑truth permutation matrix:



\[
\text{labels} = \arg\max_j (P_{\text{gt}})_{i,j}
\]



This enables GNNExplainer to attribute predictions to specific A‑graph structures.

---

# What GNNExplainer Reveals

For each S‑graph node, GNNExplainer identifies:
- Influential A‑graph nodes  
- Important edges  
- Feature dimensions  
- Incorrect match explanations  
- Missing ground‑truth match indicators  

---

# Visualization Legend

| Element | Meaning |
|--------|---------|
| **Colorbar (🔴→🔵)** | Node/edge importance |
| **Gold dot** | S‑node being explained |
| **Green line** | Correct predicted match |
| **Red line** | Incorrect predicted match |
| **Orange dashed line** | Missing ground‑truth match |

---

# 💻 PyG Implementation Snippet

```python
explainer = Explainer(
    model=classifier,
    algorithm=GNNExplainer(epochs=100, lr=0.01),
    explanation_type='model',
    model_config=dict(
        mode='multiclass_classification',
        task_level='node',
        return_type='raw',
    ),
    node_mask_type='attributes',
    edge_mask_type='object',
)
```

---

# Hyperparameter Optimization (Optuna)

The notebook includes:
- Search space definitions  
- Objective function  
- Automated trial logging  
- Best‑model extraction  

---

# References

- Ying et al. (2019) — *GNNExplainer: Generating Explanations for Graph Neural Networks*  
  https://arxiv.org/abs/1903.03894  
- PyTorch Geometric Documentation  
  https://pytorch-geometric.readthedocs.io/en/latest/modules/explain.html
