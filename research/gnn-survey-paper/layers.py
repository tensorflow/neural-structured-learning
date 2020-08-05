import tensorflow as tf

class GraphConvLayer(tf.keras.layers.Layer):
    """ Single graph convolution layer

    Args:
        input_dim (int): Input dimension of gcn layer
        output_dim (int): Output dimension of gcn layer
        bias (bool): Whether bias needs to be added to the layer
    """
    def __init__(self, **kwargs):
        super(GraphConvLayer, self).__init__()
        for key, item in kwargs.items():
            setattr(self, key, item)
        
    def build(self, input_shape):
        self.weight = self.add_weight(name="weight",
                                      shape=(self.input_dim, self.output_dim),
                                      initializer='random_normal',
                                      trainable=True)
        if self.bias:
            self.b = self.add_weight(name="bias",
                                        shape=(self.output_dim,),
                                        initializer='random_normal',
                                        trainable=True)
    def call(self, inputs):
        x, adj = inputs[0], inputs[1]
        x = tf.matmul(adj, x)
        outputs = tf.matmul(x, self.weight)
        if self.bias:
            return self.b + outputs
        else:
            return outputs

