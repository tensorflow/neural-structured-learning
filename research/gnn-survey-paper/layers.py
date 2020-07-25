import tensorflow as tf

class GraphConvLayer(tf.keras.layers.Layer):

    def __init__(self, input_dim, units):
        super(GraphConvLayer, self).__init__()
        self.weight = self.add_weight(name="weight",
                                      shape=(input_dim, units),
                                      trainable=True)    
    def call(self, inputs):
        x, adj = inputs[0], inputs[1]
       	x = tf.matmul(adj, x)
        outputs = tf.matmul(x, self.weight)
        return outputs 
        
