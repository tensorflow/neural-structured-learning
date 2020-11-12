# Graph Neural Networks (GNN) models

This repository contains a tensorflow 2.0 implementation of GNN models for node
classification.

Update: ** Sparse version of GCN and GAT are available **

## Code organization

*   `download_dataset.sh`: Download graph dataset. Now only `cora` is available.

*   `train.py`: Trains a model with FLAGS parameters. `python train.py --help`
    for more information.

*   `models.py`: Gnn models implementation. Now `gcn` and `gat` are available.

*   `layers.py`: Single gnn layers implementation.

*   `utils.py`: Data processing utils functions.

## Code usage

1.  Install required libraries, e.g. `pip install tensorflow`

2.  Download the dataset.

```
bash download_dataset.sh
```

1.  Train GAT on cora with default parameters.

```
SAVE_DIR="models/cora/gat"
python train.py --save_dir=$SAVE_DIR
```

1.  Check test results after training.

## Training Results

*   Better GAT results on cora (84.7% average test accuracy)[^GAT]:

```
python train.py \
  --model=gat \
  --gpu=0 \
  --epochs=500 \
  --lr=0.005 \
  --weight_decay=5e-4 \
  --dropout_rate=0.6 \
  --hidden_dim=8 \
  --num_heads=8 \
  --save_dir=models/cora/gat \
  --normalize_adj=True \
  --sparse_features=True
```

*   Reproduce gcn results on cora (81.5% average test accuracy)[^GCN]:

```
python train.py \
  --model=gcn \
  --gpu=0 \
  --epochs=300 \
  --lr=0.01 \
  --weight_decay=5e-4 \
  --dropout_rate=0.5 \
  --hidden_dim=16 \
  --save_dir=models/cora/gcn \
  --normalize_adj=True \
  --sparse_features=True
```

*   Better gcn results on cora (82.5% average test accuracy):

```
python train.py \
  --model=gcn \
  --gpu=0 \
  --epochs=500 \
  --lr=0.01 \
  --weight_decay=5e-4 \
  --dropout_rate=0.6 \
  --hidden_dim=32 \
  --save_dir=models/cora/gcn \
  --normalize_adj=True \
  --sparse_features=True
```

*   GIN results on cora (81.7% average test accuracy):

```
python train.py \
  --model=gin \
  --gpu=0 \
  --epochs=150 \
  --lr=0.01 \
  --weight_decay=5e-4 \
  --dropout_rate=0.8 \
  --hidden_dim=64 \
  --save_dir=models/cora/gin \
  --learn_eps=False \
  --normalize_adj=False \
  --sparse_features=True
```

## References

[^GCN]: Thomas N. Kipf, Max Welling "Semi-Supervised Classification with Graph
    Convolutional Networks"
    [GCN original github](https://github.com/tkipf/gcn/tree/master/gcn)
[^GAT]: Petar Veličković, Guillem Cucurull, et al. "Graph Attention Networks"
    [GAT original github](https://github.com/PetarV-/GAT)
