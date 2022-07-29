# Release 1.4.0

## Major Features and Improvements

*   Add `params` as an optional third argument to the `embedding_fn` argument of
    `nsl.estimator.add_graph_regularization`. This is similar to the `params`
    argument of an Estimator's `model_fn`, which allows users to pass arbitrary
    states through. Adding this as an argument to `embedding_fn` will allow
    users to access that state in the implementation of `embedding_fn`.
*   Both `nsl.keras.AdversarialRegularization` and
    `nsl.keras.GraphRegularization` now support the `save` method which will
    save the base model.
*   `nsl.keras.AdversarialRegularization` now supports a `tf.keras.Sequential`
    base model with a `tf.keras.layers.DenseFeatures` layer.
*   `nsl.configs.AdvNeighborConfig` has a new field `random_init`. If set to
    `True`, a random perturbation will be performed before FGSM/PGD steps.
*   `nsl.lib.gen_adv_neighbor` now has a new parameter `use_while_loop`. If set
    to `True`, the PGD steps are done in a `tf.while_loop` which is potentially
    more memory efficient but has some restrictions.
*   New library functions:
    *   `nsl.lib.random_in_norm_ball` for generating random tensors in a norm
        ball.
    *   `nsl.lib.project_to_ball` for projecting tensors onto a norm ball.

## Bug Fixes and Other Changes

*   Dropped Python 2 support (which was deprecated 2+ years ago).
*   `nsl.keras.AdversarialRegularization` and `nsl.lib.gen_adv_neighbor` will
    not attempt to calculate gradients for tensors with a non-differentiable
    `dtype`. This doesn’t change the functionality, but only suppresses excess
    warnings.
*   Both `estimator/adversarial_regularization.py` and
    `estimator/graph_regularization.py` explicitly import `estimator` from
    `tensorflow` as a separate import instead of accessing it via `tf.estimator`
    and depend on the tensorflow `estimator` target.
*   The new top-level `workshops` directory contains presentation materials from
    tutorials we organized on NSL at KDD 2020, WSDM 2021, and WebConf 2021.
*   The new `usage.md` page describes featured usage of NSL, external talks,
    blog posts, media coverage, and more.
*   End-to-end examples under the `examples` directory:
    *   New examples about graph neural network modules with graph-regularizer
        and graph convolution.
    *   New README file providing an overview of the examples.
*   New tutorial examples under the `examples/notebooks` directory:
    *   Graph regularization for image classification using synthesized graphs
    *   Adversarial Learning: Building Robust Image Classifiers
    *   Saving and loading NSL models

## Thanks to our Contributors

This release contains contributions from many people at Google Research and from
TF community members: @angela-wang1 , @dipanjanS, @joshchang1112, @SamuelMarks,
@sayakpaul, @wangbingnan136, @zoeyz101

# Release 1.3.1

## Major Features and Improvements

None.

## Bug Fixes and Other Changes

*   Fixed the NSL graph builder to ignore `lsh_rounds` when `lsh_splits` < 1. By
    default, the prior version of the graph builder would repeat the work twice
    by default. In addition, the default value for `lsh_rounds` has been changed
    from 2 to 1.
*   Updated the NSL IMDB tutorial to use the new LSH support when building the
    graph, thereby speeding up the graph building time by ~5x.

## Thanks to our Contributors

This release contains contributions from many people at Google.

# Release 1.3.0

## Major Features and Improvements

*   Added locality-sensitive hashing (LSH) support to the graph builder tool.
    This allows the graph builder to scale up to larger input datasets. As part
    of this change, the new `nsl.configs.GraphBuilderConfig` class was
    introduced, as well as a new `nsl.tools.build_graph_from_config` function.
    The new parameters for controlling the LSH algorithm are named `lsh_rounds`
    and `lsh_splits`.

## Bug Fixes and Other Changes

*   Fixed a bug in `nsl.tools.read_tsv_graph` that was incrementing the edge
    count too often.
*   Changed `nsl.tools.add_edge` to return a boolean result indicating if a new
    edge was added or not; previously, this function was not returning any
    value.
*   Removed Python 2 unit tests.
*   Fixed a bug in `nsl.estimator.add_adversarial_regularization` and
    `nsl.estimator.add_graph_regularization` so that the `UPDATE_OPS` can be
    triggered correctly.
*   Updated graph-NSL tutorials not to parse neighbor features during
    evaluation.
*   Added scaled graph and adversarial loss values as scalars to the summary in
    `nsl.estimator.add_graph_regularization` and
    `nsl.estimator.add_adversarial_regularization` respectively.
*   Updated graph and adversarial regularization loss metrics in
    `nsl.keras.GraphRegularization` and `nsl.keras.AdversarialRegularization`
    respectively, to include scaled values for consistency with their respective
    loss term contributions.

## Thanks to our Contributors

This release contains contributions from many people at Google.

# Release 1.2.0

## Major Features and Improvements

*   Changed `nsl.tools.build_graph(...)` to be more efficient and use far less
    memory. In particular, the memory consumption is now proportional only to
    the size of the input, not the size of the input plus the size of the
    output. Since the size of the output can be quadratic in the size of the
    input, this can lead to large memory savings. `nsl.tools.build_graph(...)`
    now also produces a log message every 1M edges it writes to indicate
    progress.
*   Introduces `nsl.lib.strip_neighbor_features`, a function to remove graph
    neighbor features from a feature dictionary.
*   Restricts the expectation of graph neighbor features being present in the
    input to the training mode for both the Keras and Estimator graph
    regularization wrappers. So, during evaluation, prediction, etc, neighbor
    features need not be fed to the model anymore.
*   Change the default value of `keep_rank` from `False` to `True` as well as
    flip its semantics in `nsl.keras.layers.NeighborFeatures.call` and
    `nsl.utils.unpack_neighbor_features`.
*   Supports feature value constraints for adversarial neighbors. See
    `clip_value_min` and `clip_value_max` in `nsl.configs.AdvNeighborConfig`.
*   Supports adversarial regularization with PGD in Keras and estimator models.
*   Support for generating adversarial neighbors using Projected Gradient
    Descent (PGD) via the `nsl.lib.adversarial_neighbor.gen_adv_neighbor` API.

## Bug Fixes and Other Changes

*   Clarifies the meaning of the `nsl.AdvNeighborConfig.feature_mask` field.
*   Updates notebooks to avoid invoking the `nsl.tools.build_graph` and
    `nsl.tools.pack_nbrs` utilities as binaries.
*   Replace deprecated API in notebooks when testing for GPU availability.
*   Fix typos in documentation and notebooks.
*   Improvements to example trainers.
*   Fixed the metric string to 'acc' to be compatible with both TF1.x and 2.x.
*   Allow passing dictionaries to sequential base models in adversarial
    regularization.
*   Supports input feature list in `nsl.lib.gen_adv_neighbor`.
*   Supports input with a collection of tensors in
    `nsl.lib.maximize_within_unit_norm`.
*   Adds an optional parameter `base_with_labels_in_features` to
    `nsl.keras.AdversarialRegularization` for passing label features to the base
    model.
*   Fixes the tensor ordering issue in `nsl.keras.AdversarialRegularization`
    when used with a functional Keras base model.

## Thanks to our Contributors

This release contains contributions from many people at Google as well as
@mzahran001.

# Release 1.1.0

## Major Features and Improvements

*   Introduces `nsl.tools.build_graph`, a function for graph building.

*   Introduces `nsl.tools.pack_nbrs`, a function to prepare input for
    graph-based NSL.

*   Adds `tf.estimator.Estimator` support for NSL. In particular, this release
    introduces two new wrapper functions named
    `nsl.estimator.add_graph_regularization` and
    `nsl.estimator.add_adversarial_regularization` to wrap existing
    `tf.estimator.Estimator`-based models with NSL. These APIs are currently
    supported only for TF 1.x.

## Bug Fixes and Other Changes

*   Adds version information to the NSL package, which can be queried as
    `nsl.__version__`.

*   Fixes loss computation with `Loss` objects in `AdversarialRegularization`.

*   Adds a new parameter to `nsl.keras.adversarial_loss` which can be used to
    pass additional arguments to the model.

*   Fixes typos in documentation and notebooks.

*   Updates notebooks to use the release version of TF 2.0.

## Thanks to our Contributors

This release contains contributions from many people at Google.

# Release 1.0.1

## Major Features and Improvements

*   Adds 'make_graph_reg_config', a new API to help construct a
    `nsl.configs.GraphRegConfig` object.

*   Updates the package description on PyPI.

## Bug Fixes and Other Changes

*   Fixes metric computation with `Metric` objects in
    `AdversarialRegularization`.

*   Fixes typos in documentation and notebooks.

## Thanks to our Contributors

This release contains contributions from many people at Google, as well as:

@joaogui1, @aspratyush.

# Release 1.0.0

*   Initial release of Neural Structured Learning in TensorFlow.
