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
