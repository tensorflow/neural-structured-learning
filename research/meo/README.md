# Manipulating embeddings with obfuscations.

## Description

This codebase provides some techniques to create robust embeddings under
various obfuscations. The code provided here robustifies the embeddings
themselves, without fine tuning the rest of the model. The intent for this
is to train models which are robust to obfuscations, without the need of
retraining a very large architecture from scratch.

The approach taken for this in this repository is done by generating
extra obfuscated embeddings. These embeddings are trained so that they mimic the
real obfuscated embeddings of each image, for any given obfuscation type.
These embeddings are then used as extra data, to train a downstream classifier.
Modeling these obfuscated embeddings is intended to help the model later
on classify images under unseen obfuscations. More specifically, the generated
obfuscated embeddings are used as additional training data when training a
classifier.

## Methods

The files in this repository cover two basic methods:

- ```multiple_decoders.py```: This trains a model using an autoencoder style
architecture, with one decoder per obfuscation type. This model receives a
clean embedding as input, and generates a corresponding obfuscated embedding
for each obfuscation type. This allows the model to be more flexible,
as there is a separate portion dedicated to each obfuscation type.

- ```parameter_generator.py```: This trains a model using an autoencoder
style architecture, where the decoder is not trained, but rather its
parameters are produced by a different architecture, which is trained. The
latter receives as input the obfuscation type, and provides as output the
parameters of the decoder corresponding to each seen obfuscation.

Finally, ```linear_finetuning.py``` is provided, which trains only a linear
classifier on top of frozen embeddings. A sample run command for this is the
following:

## Auxiliary files:

- ```configs.py```: File containing metadata for the dataset and the models
used.

- ```extended_model.py```: File containing architecture definitions for our
  models.

- ```losses.py```: File containing the losses for our models.

- ```obfuscations.py```: File containing definitions for the datasets that
we use.

## Data required

The provided code can receive data in two formats for the parameter
```data_dir_train``` (directory of data to be used during training):

- In the case of ```input_feature_name==pixel```, the data is assumed to be
in the format of ```tf.train.Example``` protos, where each field has a key
named ```label```, and one key of the form ```image_{obf}```, for each
obfuscation ```obf``` in the set of valid obfuscations.

- In the case of ```input_feature_name==embed```, the data is assumed to be
in the format of ```tf.train.Example``` protos, with a key named ```label```
containing the label of the image and a key named ```embed```, containing a
matrix of size $N \times d$, where $N$ is the number of obfuscations and
$d$ is the dimension of the embedding.

Contributor: Georgios Smyrnis
