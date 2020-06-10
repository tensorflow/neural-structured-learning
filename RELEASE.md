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
    `nsl.utils.unpack_neighbor_features`
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
