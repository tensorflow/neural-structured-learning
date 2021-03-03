Example to run CARLS

```script
bash examples/preprocess/cora/prep_data.sh

bazel run research/carls/examples:graph_keras_mlp_cora -- \
/tmp/cora/train_merged_examples.tfr /tmp/cora/test_examples.tfr \
--alsologtostderr --output_dir=/tmp/carls
```
