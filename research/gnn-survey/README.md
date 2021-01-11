# Graph Neural Networks (GNN) models

This repository contains a tensorflow 2.0 implementation of GNN models for node
classification.

Update: ** Sparse version of GCN, GAT, GIN are available **

## Code organization

*   `download_dataset.sh`: Download graph dataset. Now `cora` and `citeseer` are
    available.

*   `train.py`: Trains a model with FLAGS parameters. `python train.py --help`
    for more information.

*   `models.py`: Gnn models implementation. Now `gcn`, `gat` and `gin` are
    available.

*   `layers.py`: Single gnn layers implementation.

*   `utils.py`: Data processing utils functions.

## Code usage

1.  Install required libraries, e.g. `pip install tensorflow`

2.  Download the dataset.

```
bash download_dataset.sh <DATASET>
```

1.  Train GAT on cora with default parameters.

```
SAVE_DIR="models/cora/gat"
python train.py --save_dir=$SAVE_DIR
```

1.  Check test results after training.

## Training Results

*   Better GAT results on cora (84.7% average test accuracy,
    [[2]](#references)):

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

*   Reproduce gcn results on cora (81.5% average test accuracy,
    [[1]](#references)):

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

*   Better gcn results on cora (82.5% average test accuracy,
    [[1]](#references)):

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

*   GIN results on cora (81.7% average test accuracy, [[3]](#references)):

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

[[1] T. Kipf and M. Welling. "Semi-Supervised Classification with Graph
Convolutional Networks" ICLR 2017](https://arxiv.org/pdf/1609.02907.pdf)

[[2] P. Veličković, G. Cucurull, A. Casanova, A. Romero, P. Liò and Y. Bengio.
"Graph Attention Networks" ICLR 2018](https://arxiv.org/pdf/1710.10903.pdf)

[[3] K. Xu, W. Hu, J. Leskovec and S. Jegelka. "How Powerful are Graph Neural
Networks?" ICLR 2019](https://arxiv.org/pdf/1810.00826.pdf)
