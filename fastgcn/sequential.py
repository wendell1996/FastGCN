from .metrics import *
from .utils import *
import tensorflow as tf
import fastgcn.layers

def iterate_minibatches_listinputs(inputs, batchsize, shuffle=False):
    assert inputs is not None
    numSamples = inputs[0].shape[0]
    if shuffle:
        indices = np.arange(numSamples)
        np.random.shuffle(indices)
    for start_idx in range(0, numSamples - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield [input[excerpt] for input in inputs]

def minibatches(inputs,batchsize,shuffle=False):
    assert inputs is not None
    numSamples = inputs[0].shape[0]
    ans = []
    if shuffle:
        indices = np.arange(numSamples)
        np.random.shuffle(indices)
    for start_idx in range(0, numSamples - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        ans.append([input[excerpt] for input in inputs])
    return ans

class Sequential(object):
    def __init__(self,**kwargs):
        allowed_kwargs = {"name"}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name
        self.vars = {}
        self.layers = []
        self.inputs = None
        self.outputs = None
        self.activations = []
        self.epochs = 0
        self.sample_num = 0
        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None
        self.metrics = None
        self.input_shape = None
        self.support = None
        self.gc_layer_indices = []
        self.layers_num = 0

    def _loss(self,losses,weight_decay):
        for loss in losses:
            if loss == "l2_loss":
                for var in self.layers[0].vars.values():
                    self.loss += weight_decay * tf.nn.l2_loss(var)
            if loss == "softmax_cross_entropy":
                self.loss += softmax_cross_entropy(self.outputs,self.placeholders["labels"])

    def _accuracy(self,metrics):
        for metric in metrics:
            if metric == "accuracy":
                self.accuracy = accuracy(self.outputs,self.placeholders["labels"])

    def add(self, layer):
        if isinstance(layer,fastgcn.layers.GraphConvolution):
            self.gc_layer_indices.append(self.layers_num)
        with tf.variable_scope(self.name):
            self.layers.append(layer)
        self.layers_num += 1

    def compile(self,
                optimizer=None,
                losses=["softmax_cross_entropy","l2_loss"],
                weight_decay = 5e-4,
                learning_rate=0.0001,
                metrics=["accuracy"],
                **kwargs):
        self.metrics = metrics or []
        self.input_shape = self.layers[0].input_shape
        self.placeholders = {
            "supports": [tf.placeholder(name="support_"+str(i),dtype=tf.float32) for i in range(len(self.gc_layer_indices))],
            "inputs": tf.placeholder(name="inputs",dtype=tf.float32,shape=[None,self.input_shape[1]]),
            "labels": tf.placeholder(name="labels",dtype=tf.float32)
        }
        self.activations.append(self.placeholders["inputs"])
        support = None
        n = len(self.gc_layer_indices) - 1
        for i,layer in enumerate(self.layers):
            if i in self.gc_layer_indices:
                support = self.placeholders["supports"][n]
                n -= 1
            hidden = layer(inputs=self.activations[-1],support=support)
            support = None
            self.activations.append(hidden)
        self.outputs = self.activations[-1]
        # self.labels = tf.placeholder(shape=[None,self.outputs.shape[1]],name="labels",dtype=tf.float32)

        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        self._loss(losses=losses,weight_decay=weight_decay)

        if optimizer == None:
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        else:
            self.optimizer = optimizer
        self._accuracy(metrics)

        self.train_step = self.optimizer.minimize(self.loss)

    def fit(self,
            x=None,
            y=None,
            batch_size=None,
            epochs=1,
            ranks=None,
            placeholders=None,
            **kwargs):
        inputs = x
        labels = y
        self.epochs = epochs
        self.ranks = ranks
        adjacent_matrix_train = placeholders["supports"][0]
        assert np.sum(adjacent_matrix_train)
        p = np.sum(adjacent_matrix_train,axis=0) / np.sum(adjacent_matrix_train)
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        for epoch in range(self.epochs):
            batch_num = 0
            start = time.time()
            batches = iterate_minibatches_listinputs([adjacent_matrix_train, labels], batchsize=batch_size, shuffle=False)
            for batch in batches:
                [adjacent_matrix_train_batch, labels_train_batch] = batch
                supports = []
                if self.ranks is None:
                    #for sparse matrix
                    pass
                else:
                    adjacent_matrix_train_batch_temp = adjacent_matrix_train_batch
                    for i,gc_index in enumerate(self.gc_layer_indices):
                        nonzero_in_degree_vector = np.nonzero(np.sum(adjacent_matrix_train_batch_temp, axis=0))[0]
                        assert len(nonzero_in_degree_vector) > 0,"invaild nonzero"
                        if self.ranks[-i-1] > len(nonzero_in_degree_vector):
                            rank_temp = len(nonzero_in_degree_vector)
                        else:
                            rank_temp = self.ranks[-i-1]
                        q = np.random.choice(nonzero_in_degree_vector, rank_temp, replace=False, p=p[nonzero_in_degree_vector] / sum(p[nonzero_in_degree_vector]))
                        if i ==0:
                            supports.append(np.dot(adjacent_matrix_train_batch_temp[:, q],np.eye(len(q))*(1.0 / (p[q] * rank_temp))))
                        else:
                            supports.append(adjacent_matrix_train_batch_temp[:, q])
                        adjacent_matrix_train_batch_temp = adjacent_matrix_train[q,:]
                    inputs_batch = inputs[q,:]
                    feed_dict = construct_feed_dict(inputs=inputs_batch,labels=labels_train_batch,supports=supports,placeholders=self.placeholders,len=len(self.gc_layer_indices))
                _,loss,accuracy = self.session.run([self.train_step, self.loss, self.accuracy], feed_dict=feed_dict)
                # print("%s %s(%s seconds)" % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                #                                 "epoch " + str(epoch + 1)
                #                                 + " |batch " + str(batch_num + 1)
                #                                 + " |loss " + str(loss)
                #                                 + " |accuracy " + str(accuracy),
                #                                 round(time.time() - start, 3)))
                batch_num += 1
            supports = []
            for i in range(len(self.gc_layer_indices)):
                supports.append(adjacent_matrix_train)
            feed_dict = construct_feed_dict(inputs=inputs,labels=labels,supports=supports,placeholders=self.placeholders, len=len(self.gc_layer_indices))
            _, loss, accuracy = self.session.run([self.train_step, self.loss, self.accuracy], feed_dict=feed_dict)
            print("%s %s(%s seconds)" % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                            "epoch " + str(epoch + 1)
                                            + " |loss " + str(loss)
                                            + " |accuracy " + str(accuracy),
                                            round(time.time() - start, 3)))

    def evaluate(self,
                 x=None,
                 y=None,
                 placeholders=None):
        supports = placeholders['supports']
        start = time.time()
        feed_dict = construct_feed_dict(inputs=x, labels=y,supports=supports,placeholders=self.placeholders,len=len(self.gc_layer_indices))
        _,loss,accuracy = self.session.run([self.train_step, self.loss, self.accuracy],feed_dict=feed_dict)
        if self.ranks == None:
            print("%s %s(%s seconds)" % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                         "loss " + str(loss)
                                         + " |accuracy " + str(accuracy),
                                         round(time.time() - start, 3)))
        else:
            print("%s" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                         *map(lambda x:x,["rank"+str(i)+"="+str(rank) for i,rank in enumerate(self.ranks)]),
                          "%s (%s seconds)" % ("|loss " + str(loss)
                                         + " |accuracy " + str(accuracy),
                                         round(time.time() - start, 3)))