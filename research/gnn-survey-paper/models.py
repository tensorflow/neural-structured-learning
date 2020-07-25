import tensorflow as tf
from layers import GraphConvLayer

class GCN(tf.keras.Model):

    def __init__(self, features_dim, num_layers, hidden_dim, num_classes, dropout_rate):
        super(GCN, self).__init__()
        
        self.num_layers = num_layers

        self.gc = []
        # input layer 
        single_gc = tf.keras.Sequential()
        single_gc.add(GraphConvLayer(features_dim, hidden_dim[0]))
        single_gc.add(tf.keras.layers.ReLU())
        single_gc.add(tf.keras.layers.Dropout(dropout_rate))
        self.gc.append(single_gc)
        
        # hidden layers
        for i in range(0, num_layers-2):
            single_gc = tf.keras.Sequential()
            single_gc.add(GraphConvLayer(hidden_dim[i], hidden_dim[i+1]))
            single_gc.add(tf.keras.layers.ReLU())
            single_gc.add(tf.keras.layers.Dropout(dropout_rate))
            self.gc.append(single_gc)
        
        # output layer
        self.classifier = GraphConvLayer(hidden_dim[-1], num_classes)
        
    def call(self, features, adj):

        for i in range(self.num_layers-1):
            x = (features, adj) 
            features = self.gc[i](x)
        
        x = (features, adj)
        outputs = self.classifier(x)
        return outputs

