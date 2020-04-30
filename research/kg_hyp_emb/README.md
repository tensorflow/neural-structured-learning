# Knowledge Graph (KG) Embedding Library

This project is a Tensorflow 2.0 implementation of Hyperbolic KG embeddings [6]
as well as multiple state-of-the-art KG embedding models which can be trained
for the link prediction task. A PyTorch implementation is also available at: [https://github.com/HazyResearch/KGEmb](https://github.com/HazyResearch/KGEmb)

## Library Overview

This implementation includes the following models:

#### Complex embeddings:

*   Complex [1]
*   Complex-N3 [2]
*   RotatE [3]

#### Euclidean embeddings:

*   CTDecomp [2]
*   TransE [4]
*   MurE [5]
*   RotE [6]
*   RefE [6]
*   AttE [6]

#### Hyperbolic embeddings:

*   TransH [6]
*   RotH [6]
*   RefH [6]
*   AttH [6]

## Installation

First, create a python 3.7 environment and install dependencies: From kgemb/

```bash
virtualenv -p python3.7 kgenv
```

```bash
source kgenv/bin/activate
```

```bash
pip install -r requirements.txt
```

Then, download and pre-process the datasets:

```bash
source datasets/download.sh
```

```bash
python datasets/process.py
```

Add the package to your local path:

```bash
KG_DIR=$(pwd)/..
```

```bash
export PYTHONPATH="$KG_DIR:$PYTHONPATH"
```

## Example usage

Then, train a model using the `train.py` script. We provide an example to train
RefE on FB15k-237:

```bash
python train.py --max_epochs 100 --dataset FB237 --model RefE --loss_fn SigmoidCrossEntropy --neg_sample_size -1 --data_dir data --optimizer Adagrad --lr 5e-2 --save_dir logs --rank 500 --entity_reg 1e-5 --rel_reg 1e-5 --patience 10 --valid 5 --save_model=false --save_logs=true --regularizer L3 --initializer GlorotNormal
```

This model achieves 54% Hits@10 on the FB237 test set.

## New models

To add a new (complex/hyperbolic/Euclidean) Knowledge Graph embedding model, implement the corresponding query embedding under models/, e.g.:

```
def get_queries(self, input_tensor):
    entity = self.entity(input_tensor[:, 0])
    rel = self.rel(input_tensor[:, 1])
    result = ### Do something here ###
    return return result
```

## Citation

If you use the codes, please cite the following paper [6]:

```
TODO: add bibtex
```

## References

[1] Trouillon, Théo, et al. "Complex embeddings for simple link prediction."
International Conference on Machine Learning. 2016.

[2] Lacroix, Timothee, et al. "Canonical Tensor Decomposition for Knowledge Base
Completion." International Conference on Machine Learning. 2018.

[3] Sun, Zhiqing, et al. "Rotate: Knowledge graph embedding by relational
rotation in complex space." International Conference on Learning
Representations. 2019.

[4] Bordes, Antoine, et al. "Translating embeddings for modeling
multi-relational data." Advances in neural information processing systems. 2013.

[5] Balažević, Ivana, et al. "Multi-relational Poincaré Graph Embeddings."
Advances in neural information processing systems. 2019.

[6] Chami, Ines, et al. "Low-Dimensional Hyperbolic Knowledge Graph Embeddings." Annual  Meeting  of  the  Association  for Computational Linguistics. 2020.

