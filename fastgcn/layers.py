from .inits import *

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}

def sparse_dropout(x,keep_prob,noise_shape):
    """Dropout for sparse tensors."""
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)


def dot(x,y,sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res

def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]

class Layer(object):
    """
    Implementation inspired by GCN and Keras
    """
    def __init__(self,**kwargs):
        allowed_kwargs = {
            'input_shape',
            'dtype',
            'name',
            'trainable',
            'weights'
        }
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs,"Invaild keyword argument" + kwarg
        self.input_shape = kwargs.get("input_shape")
        self.dtype = kwargs.get("dtype")
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name

    def _call(self, inputs):
        return inputs

    def __call__(self,inputs,support):
        outputs = self._call(inputs,support)
        return outputs

class GraphConvolution(Layer):
    def __init__(self,
                 output_dim,
                 placeholders=None,
                 sparse_inputs=False,
                 bias=False,
                 featureless=False,
                 activation=tf.nn.relu,
                 dropout=0.,
                 **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)

        self.placeholders = placeholders

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.
        self.activation = activation
        # if support is None:
        #     self.support = placeholders['support']
        # else:
        #     self.support = support
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias
        self.output_dim = output_dim[0]
        self.input_dim = None

        # helper variable for sparse dropout
        if sparse_inputs:
            self.num_features_nonzero = placeholders['num_features_nonzero']

        self.vars = {}

    def _call(self,inputs,support):
        x = inputs
        self.input_dim = x.shape.as_list()[1]
        with tf.variable_scope(self.name + '_vars'):
            for i in range(1):
                self.vars['weights_' + str(i)] = glorot([self.input_dim, self.output_dim],name='weights_' + str(i))
            if self.bias:
                self.vars['bias'] = zeros([self.output_dim], name='bias')
        if self.sparse_inputs:
            x = sparse_dropout(x, 1 - self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1 - self.dropout)

        if not self.featureless:
            pre_support = dot(x, self.vars['weights_0'],sparse=self.sparse_inputs)
        else:
            pre_support = self.vars['weights_0']
        output = dot(support, pre_support, sparse=False)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.activation(output)

class Dense(Layer):
    def __init__(self,
                 output_dim,
                 placeholders=None,
                 dropout=0.,
                 sparse_inputs=False,
                 activation=tf.nn.relu,
                 bias=False,
                 featureless=False,
                 **kwargs):
        super(Dense,self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.activation = activation
        self.input_dim = self.input_shape[1]
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        if sparse_inputs:
            self.num_features_nonzero = placeholders['num_features_nonzero']

        self.vars = {}
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = glorot([self.input_dim,output_dim],name='weights')
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

    def _call(self,inputs,support):
        x = inputs

        if self.sparse_inputs:
            x = sparse_dropout(x, 1 - self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1 - self.dropout)

        output = dot(x,self.vars['weights'],sparse=self.sparse_inputs)

        if self.bias:
            output += self.vars['bias']
        return self.activation(output)