import tensorflow as tf
from layers import GraphConvLayer

class GCN(tf.keras.Model):
    """Graph convolution network for semi-supevised node classification.

    Args:
        features_dim (int): Dimension of input features 
        num_layers (int): Number of gnn layers 
        hidden_dim (list): List of hidden layers dimension
        num_classes (int): Total number of classes
        dropout_prob (float): Dropout probability
        bias (bool): Whether bias needs to be added to gcn layers
    """
    
    def __init__(self, **kwargs):
        super(GCN, self).__init__()
        
        for key, item in kwargs.items():
            setattr(self, key, item)
        
        self.gc = []
        # input layer
        single_gc = tf.keras.Sequential()
        single_gc.add(GraphConvLayer(input_dim=self.features_dim,
                                     output_dim=self.hidden_dim[0],
                                     bias=self.bias))
        single_gc.add(tf.keras.layers.ReLU())
        single_gc.add(tf.keras.layers.Dropout(self.dropout_prob))
        self.gc.append(single_gc)

        # hidden layers
        for i in range(0, self.num_layers-2):
            single_gc = tf.keras.Sequential()
            single_gc.add(GraphConvLayer(input_dim=self.hidden_dim[i],
                                         output_dim=self.hidden_dim[i+1],
                                         bias=self.bias))
            single_gc.add(tf.keras.layers.ReLU())
            single_gc.add(tf.keras.layers.Dropout(self.dropout_prob))
            self.gc.append(single_gc)

        # output layer
        self.classifier = GraphConvLayer(input_dim=self.hidden_dim[-1],
                                         output_dim=self.num_classes,
                                         bias=self.bias)

    def call(self, inputs):
        features, adj = inputs[0], inputs[1]
        for i in range(self.num_layers-1):
            x = (features, adj)
            features = self.gc[i](x)

        x = (features, adj)
        outputs = self.classifier(x)
        return outputs
