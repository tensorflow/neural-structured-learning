import tensorflow as tf

epsilon = 5
num_power_iterations = 1
xi = 1e-6
scale_r = False


def kl_divergence_with_logit(q_logit, p_logit):
    q = tf.nn.softmax(q_logit)
    qlogq = -tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=q, logits=q_logit)
    qlogp = -tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=q, logits=p_logit)
    return qlogq - qlogp


def get_normalized_vector(d):
    d /= (1e-12 + tf.reduce_max(tf.abs(d), keep_dims=True))
    d /= tf.sqrt(1e-6 + tf.reduce_sum(tf.pow(d, 2.0), keep_dims=True))
    return d


def get_normalizing_constant(d):
    c = 1e-12 + tf.reduce_max(tf.abs(d), keep_dims=True)
    c *= tf.sqrt(1e-6 + tf.reduce_sum(tf.pow(d, 2.0), keep_dims=True))
    return c


def get_loss_vat(inputs, predictions, is_train, model, predictions_var_scope):
    r_vadv = generate_virtual_adversarial_perturbation(
        inputs, predictions, model, predictions_var_scope, is_train=is_train)
    predictions = tf.stop_gradient(predictions)
    logit_p = predictions
    new_inputs = tf.add(inputs, r_vadv)
    with tf.variable_scope(
        predictions_var_scope,  auxiliary_name_scope=False, reuse=True):
        encoding_m, _, _ = model.get_encoding_and_params(
            inputs=new_inputs,
            is_train=is_train,
            update_batch_stats=False)
        logit_m, _, _ = model.get_predictions_and_params(
            encoding=encoding_m,
            is_train=is_train)
    loss = kl_divergence_with_logit(logit_p, logit_m)
    return tf.reduce_mean(loss)


def generate_virtual_adversarial_perturbation(
        inputs, logits, model, predictions_var_scope, is_train=True):
    """Generates an adversarial perturbation for virtual adversarial training.

    Args:
        inputs: A batch of input features, where the batch is the first
            dimension.
        logits: The logits predicted by a model on the provided inputs.
        model: The model that generated the logits.
        predictions_var_scope: Variable scope for obtaining the predictions.
        is_train: A boolean placeholder specifying if this is a training or
            testing setting.

    Returns:
        A Tensor of the same shape as the inputs containing the adversarial
        perturbation for these inputs.
    """
    d = tf.random_normal(shape=tf.shape(inputs))

    for _ in range(num_power_iterations):
        d = xi * get_normalized_vector(d)
        logit_p = logits
        with tf.variable_scope(
                predictions_var_scope,  auxiliary_name_scope=False, reuse=True):
            encoding_m, _, _ = model.get_encoding_and_params(
                inputs=d + inputs,
                is_train=is_train,
                update_batch_stats=False)
            logit_m, _, _ = model.get_predictions_and_params(
                encoding=encoding_m,
                is_train=is_train)
        dist = kl_divergence_with_logit(logit_p, logit_m)
        grad = tf.gradients(dist, [d], aggregation_method=2)[0]
        d = tf.stop_gradient(grad)

    r_vadv = get_normalized_vector(d)
    if scale_r:
        r_vadv *= get_normalizing_constant(inputs)
    r_vadv *= epsilon
    return r_vadv


def logsoftmax(x):
    """Implementation of softmax when the inputs are logits."""
    xdev = x - tf.reduce_max(x, 1, keep_dims=True)
    lsm = xdev - tf.log(tf.reduce_sum(tf.exp(xdev), 1, keep_dims=True))
    return lsm


def entropy_y_x(logit):
    """Entropy term to add to VATENT."""
    p = tf.nn.softmax(logit)
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=p, logits=logit))
