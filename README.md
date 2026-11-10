# GAT model for the fraud paper

Part of a group remake of "Scalable and interpretable fraud detection in FinTech: a hybrid TabNet-Graph neural network framework" on IEEE-CIS. This folder holds the **GAT** model.

**GAT stays graph-only: it uses no tabular features.** The fusion step does the combining, using the score files this notebook saves.

## Status
`gat_fraud.ipynb` has been run on Colab GPU. All 6 runs finished (12 epochs each, no early stopping) and all 18 output files are saved under `results/` in this repo. See **Results** below.

## Data (Google Drive)
`/content/drive/MyDrive/fraud_project/data/graph_files`

| File | Content |
|---|---|
| `graph_info.json` | entity columns, node counts, missing/unknown node ids |
| `node_type.npy` | node type per node, shape [14239] |
| `graph_edges.npy` | real graph, int64 [2, 336236], both directions, no self-loops |
| `graph_shuf_seed{1,2,42}_edges.npy` | shuffled-graph controls |
| `payment_nodes_{train,val,test}.parquet` | `TransactionID`, `isFraud`, then the node id of each entity column |

Also included under `data/`: `ieee_sorted_split.parquet`, the team's shared train/val/test split (590,540 payments, sorted by `TransactionDT`, 70/15/15) that this graph was built from. Everyone on the team loads this same file rather than building their own split.

## Graph and pipeline
- Entity-only graph: 14,239 nodes of 6 types (card1, addr1, P_emaildomain, R_emaildomain, DeviceInfo, DeviceType). **Payments are not nodes.** Each payment row stores the 6 entity node ids it touches.
- Graph built from the train split only (413,378 payments).
- Shuffled graphs: each entity column's training values permuted independently per seed, links rebuilt. Same payments, different entity-entity structure. They serve as the control.

## Model
- Learnable node embedding + node-type embedding (64-d), 2 GATConv layers (4 heads, hidden 64, then 1 head), then a payment head: concatenate the 6 entity embeddings, MLP (128, then 1 logit).
- Loss: BCE with `pos_weight = sqrt(non-fraud / fraud)` (about 5.24).
- Optimizer: Adam, lr 1e-3. Batch size 4096.
- **Unknown-node replacement (training only):** each entity of a payment is replaced by its column's "unknown" node with probability 0.10. Not applied at validation or test.
- **Exactly 12 epochs, no early stopping.** The epoch with the best validation ROC-AUC is kept and used to score validation and test.
- Threshold chosen on validation (max F1), applied to test, for the precision/recall/F1 report.

## Runs (6)
The seed is set right before each model is created.

| Run | Seed | Graph file |
|---|---|---|
| `gat_seed42` | 42 | `graph_edges.npy` |
| `gat_seed1` | 1 | `graph_edges.npy` |
| `gat_seed2` | 2 | `graph_edges.npy` |
| `gat_shuf_seed42` | 42 | `graph_shuf_seed42_edges.npy` |
| `gat_shuf_seed1` | 1 | `graph_shuf_seed1_edges.npy` |
| `gat_shuf_seed2` | 2 | `graph_shuf_seed2_edges.npy` |

## Outputs (`results/`, also written to Drive at `/content/drive/MyDrive/fraud_project/results/`)
- `<run>_val.parquet`, `<run>_test.parquet`: columns `TransactionID`, `isFraud`, `score`. The score is the model's raw output (logit, before the sigmoid) from the best epoch. 12 files.
- `<run>_history.json`: run name, seed, graph file, settings, chosen epoch, and per epoch the training loss, validation ROC-AUC and validation PR-AUC. 6 files.
- `gat_results.csv`: test AUPRC, AUROC, precision, recall, F1 and threshold per run.

## Results

| run | graph | seed | AUPRC | AUROC | Precision | Recall | F1 | Threshold |
|---|---|---|---|---|---|---|---|---|
| gat_seed42 | real | 42 | 0.1565 | 0.7732 | 0.2629 | 0.2066 | 0.2314 | 0.4302 |
| gat_seed1 | real | 1 | 0.1472 | 0.7569 | 0.2407 | 0.2621 | 0.2509 | 0.3894 |
| gat_seed2 | real | 2 | 0.1400 | 0.7465 | 0.2057 | 0.2948 | 0.2423 | 0.3764 |
| gat_shuf_seed42 | shuffled | 42 | 0.1391 | 0.7332 | 0.2295 | 0.2219 | 0.2256 | 0.4553 |
| gat_shuf_seed1 | shuffled | 1 | 0.1443 | 0.7462 | 0.2373 | 0.2206 | 0.2286 | 0.4791 |
| gat_shuf_seed2 | shuffled | 2 | 0.1487 | 0.7514 | 0.2244 | 0.2400 | 0.2319 | 0.4939 |

Real-graph mean AUPRC 0.1479 vs shuffled mean 0.1440; mean AUROC 0.7589 vs 0.7436. A modest gain for the real graph, smaller than the spread across seeds — per the team plan's RQ3 criteria, this run alone does not clear the "beats shuffled by more than chance" bar on its own; it feeds into the team's bootstrap/CI analysis alongside the other models.

## Notebook conventions
Small separate cells, one step per cell. Shared cells: Model, Train function, Validate function. Each run then has Setup, Train, Best Epoch, Results, and Save cells.
