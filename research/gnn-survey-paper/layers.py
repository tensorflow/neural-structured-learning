import tensorflow as tf

class GraphConvLayer(tf.keras.layers.Layer):

    def __init__(self, features):
        super(GraphConvLayer, self).__init__()
        self.in_feat = features['input_dim']
        self.out_feat = features['output_dim']
        self.b = features['bias']

    def build(self, input_shape):
        self.weight = self.add_weight(name="weight",
                                      shape=(self.in_feat, self.out_feat),
                                      initializer='random_normal',
                                      trainable=True)
        if self.b:
            self.bias = self.add_weight(name="bias",
                                        shape=(self.out_feat,),
                                        initializer='random_normal',
                                        trainable=True)
    def call(self, inputs):
        x, adj = inputs[0], inputs[1]
       	x = tf.matmul(adj, x)
        outputs = tf.matmul(x, self.weight)
        if self.b:
            return self.bias + outputs
        else:
            return outputs

