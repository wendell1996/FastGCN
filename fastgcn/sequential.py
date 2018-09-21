from .metrics import *
from .utils import *
import tensorflow as tf

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
        allowed_kwargs = {"name","placeholders","sample_num"}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name
        self.vars = {}
        self.placeholders = kwargs.get("placeholders")
        self.layers = []
        self.inputs = None
        self.outputs = None
        self.activations = []
        self.epochs = 0
        self.sample_num = kwargs.get("sample_num")
        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None

    def _loss(self):
        for var in self.layers[0].vars.values():
            self.loss += tf.nn.l2_loss(var)
        self.loss += 10*softmax_cross_entropy(self.outputs,self.labels)

    def _accuracy(self):
        self.accuracy = accuracy(self.outputs,self.labels)

    def add(self, layer):
        with tf.variable_scope(self.name):
            self.layers.append(layer)

    def compile(self,
                optimizer=None,
                loss=None,
                metrics=["accuracy"],
                loss_weights=None,
                sample_weight_mode=None,
                weighted_metrics=None,
                target_tensors=None,
                **kwargs):
        self.metrics = metrics or []
        self.loss_weights = loss_weights
        self.sample_weight_mode = sample_weight_mode
        self.weighted_metrics = weighted_metrics
        self.input_shape = self.layers[0].input_shape
        self.inputs = tf.placeholder(shape=[None,self.input_shape[1]], name="inputs", dtype=tf.float32)
        self.support = tf.placeholder(shape=[None, self.sample_num], name="support", dtype=tf.float32)

        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(inputs=self.activations[-1],support=self.support)
            self.activations.append(hidden)
        self.outputs = self.activations[-1]
        self.labels = tf.placeholder(shape=[None,self.outputs.shape[1]],name="labels",dtype=tf.float32)

        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        if loss == None:
            self._loss()
        elif loss == "masked_softmax_cross_entropy":
            self._loss()
        else:
            raise ValueError

        if optimizer == None:
            self.optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
        else:
            self.optimizer = optimizer
        for metric in metrics:
            if metric == "accuracy":
                self._accuracy()

        self.train_step = self.optimizer.minimize(self.loss)

    def fit(self,
            x=None,
            y=None,
            batch_size=None,
            epochs=1,
            verbose=1,
            callbacks=None,
            validation_split=0.,
            validation_data=None,
            shuffle=True,
            class_weight=None,
            sample_weight=None,
            initial_epoch=0,
            steps_per_epoch=None,
            validation_steps=None,
            **kwargs):
        inputs = x
        labels = y
        self.epochs = epochs
        adjacent_matrix_train = self.placeholders["support"]
        assert np.sum(adjacent_matrix_train)
        p0 = np.sum(adjacent_matrix_train,axis=0) / np.sum(adjacent_matrix_train)
        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            for epoch in range(self.epochs):
                n = 0
                start = time.time()
                batches = minibatches([adjacent_matrix_train, labels], batchsize=batch_size, shuffle=True)
                for batch in batches:
                    [adjacent_matrix_train_batch, labels_train_batch] = batch
                    if self.sample_num is None:
                        #for sparse matrix
                        pass
                    else:
                        nonzero_in_degree_vector = np.nonzero(np.sum(adjacent_matrix_train_batch, axis=0))[0]
                        assert self.sample_num <= len(nonzero_in_degree_vector),"Invaild sample num"
                        q1 = np.random.choice(nonzero_in_degree_vector, self.sample_num, replace=False, p=p0[nonzero_in_degree_vector] / sum(p0[nonzero_in_degree_vector]))  # top layer
                        support1 = np.dot(adjacent_matrix_train_batch[:, q1],np.eye(q1.shape[0])*(1.0 / (p0[q1] * self.sample_num)))
                        inputs_batch = inputs[q1,:]
                    _,loss,accuracy = session.run([self.train_step, self.loss, self.accuracy], feed_dict={self.inputs:inputs_batch,self.labels:labels_train_batch,self.support:support1})
                    print("%s %s(%s seconds)" % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                                    "epoch " + str(epoch)
                                                    + " |batch " + str(n)
                                                    + " |loss " + str(loss)
                                                    + " |accuracy " + str(accuracy),
                                                    round(time.time() - start, 3)))
                    n = n + 1



