# GAT model for the fraud paper

## Project
Group project (4 people) remaking "Scalable and interpretable fraud detection in FinTech: a hybrid TabNet-Graph neural network framework" on IEEE-CIS. My part: the **GAT** model, using the shared graph files and pipeline.

## Deliverable
`gat_fraud.ipynb` (Colab, GPU). Has been run; results below.

## Data (Google Drive)
`/content/drive/MyDrive/fraud_project/data/graph_files`

| File | Content |
|---|---|
| `graph_info.json` | entity columns, node counts, special node ids |
| `node_type.npy` | node type per node, shape [14239] |
| `graph_edges.npy` | real graph, int64 [2, 336236], both directions, no self-loops |
| `graph_shuf_seed{1,2,42}_edges.npy` | shuffled-graph controls |
| `payment_nodes_{train,val,test}.parquet` | `TransactionID`, `isFraud`, then node id for each entity column |

The `.npy` and `.parquet` files are large and are not meant to be committed. They're in `.gitignore`.

## Graph and pipeline
- Entity-only graph: 14,239 nodes of 6 types (card1, addr1, P_emaildomain, R_emaildomain, DeviceInfo, DeviceType). **Payments are not nodes.**
- Each payment row stores the 6 entity node ids it touches.
- Graph built from the train split only (413,378 payments).
- Shuffled graphs: each entity column's training values permuted independently (per seed), links rebuilt. Same payments, different entity-entity structure. This is the control: if the real graph beats the shuffled ones, graph structure carries signal.

## Model
- Learnable node embedding + node-type embedding (64-d)
- 2 GATConv layers (4 heads, hidden 64, then 1 head)
- Payment head: concatenate the 6 entity embeddings, then MLP (128, then 1 logit)
- BCE loss with `pos_weight` for the class imbalance
- AdamW, lr 1e-3, weight decay 1e-5, up to 60 epochs, early stopping (patience 8) on validation AUPRC
- Threshold chosen on validation (max F1), applied to test
- Hyperparameters are my own defaults, not taken from the paper. Retune if teammates use different settings.
- Uses graph embeddings only, no tabular (TabNet or autoencoder) features. **Open question:** whether to add tabular features to the payment head to match how teammates fuse TabNet with the graph.

## Notebook layout and conventions
- Small, separate cells, one step per cell.
- Per model, cells in this order: Model, Train (function), Validate (function), Setup, Train, Best Epoch, Results.
- Runs: GAT on the real graph, then on shuffled seeds 1, 2 and 42, each with its own Setup, Train, Best Epoch and Results cells.
- Section 5 compares runs. Section 6 saves CSVs.

## Outputs (written to `/content/drive/MyDrive/fraud_project/results/`, also in this repo)
- `gat_results.csv`: one row per run, with test AUPRC, AUROC, Precision, Recall, F1 and Threshold
- `gat_training_history.csv`: per-epoch train loss, val loss, val AUPRC and val AUROC for every run

## Results

| model | AUPRC | AUROC | Precision | Recall | F1 | Threshold |
|---|---|---|---|---|---|---|
| GAT (real graph) | 0.1599 | 0.7714 | 0.2693 | 0.2543 | 0.2616 | 0.8293 |
| GAT (shuffled seed 1) | 0.1402 | 0.7485 | 0.2046 | 0.2644 | 0.2306 | 0.78 |
| GAT (shuffled seed 2) | 0.1318 | 0.7496 | 0.1558 | 0.2306 | 0.186 | 0.8669 |
| GAT (shuffled seed 42) | 0.1327 | 0.7473 | 0.1739 | 0.3331 | 0.2286 | 0.7287 |

The real graph beats all three shuffled controls on AUPRC, so graph structure is carrying signal.

## Next steps
1. ~~Run the notebook on Colab GPU and fix any errors.~~ Done.
2. ~~Commit the notebook (and result CSVs once they exist) to the group repo. Keep the data files out of Git.~~ Done.
3. Decide whether to add tabular features to the payment head.
