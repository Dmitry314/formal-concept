import numpy as np
import theano.tensor as T
import theano
from lasagne import *
from mldm import Net

INITIAL_SIGMA = 0.00001
EPSILON = 0.001


class SparseLayer(layers.Layer):
    def __init__(self, input_layer, C, W=None, b=None, nonlinearity=nonlinearities.rectify, **kwargs):
        super(SparseLayer, self).__init__(incoming=input_layer, **kwargs)

        if input_layer.output_shape[1] != C.shape[0]:
            print('Wrong dimensions, input_layer_size = %d, matrix_size = (%d, %d)' % (input_layer.output_shape[1],
                                                                                       C.shape[0], C.shape[1]))
            return

        if W is not None:
            if C.shape[0] != W.shape[0] or C.shape[1] != W.shape[1]:
                print('Wrong matrix dimensions, C.shape = (%d, %d), W.shape = (%d, %d)' % (C.shape[0], C.shape[1],
                                                                                       W.shape[0], W.shape[1]))
                return

        self.C = self.add_param(C, C.shape, name='C')

        if W is None:
            self.W = self.add_param(np.random.normal(loc=0, scale=INITIAL_SIGMA, size=C.shape), C.shape, name='W')
        else:
            self.W = self.add_param(W, W.shape, name='W')

        if b is None:
            self.b = None
        else:
            if C.shape[1] != b.shape[0]:
                print('Wrong bias dimension, C.shape = (%d, %d), b.shape = %d' % (C.shape[0], C.shape[1],
                                                                                  b.shape[0]))
            self.b = self.add_param(b, (C.shape[1],), name="b",
                                    regularizable=False)

        self.nonlinearity = (nonlinearities.rectify if nonlinearity is None
                             else nonlinearity)

    def get_output_shape_for(self, input_shape):
        return input_shape[:-1] + (self.C.get_value().shape[1],)

    def get_output_for(self, input, **kwargs):
        if self.b is not None:
            return self.nonlinearity(T.dot(input, self.W * self.C)+self.b)
        else:
            return self.nonlinearity(T.dot(input, self.W * self.C))


class DeepSparseNet(Net):
    def __init__(self, diagram, num_classes, W=None, nonlinearity=nonlinearities.sigmoid):
        self.X_batch = T.fmatrix(name='X_batch')
        self.y_batch = T.fmatrix(name='y_batch')

        self.layers = []
        input_l = layers.InputLayer(shape=(None,) + (diagram[0].shape[1],), input_var=self.X_batch, name='Input')
        self.layers.append(input_l)

        reshape_l = layers.FlattenLayer(input_l, name='Reshape')
        self.layers.append(reshape_l)

        if W is None:
            for i, C in enumerate(diagram):
                sparse = SparseLayer(
                    self.layers[-1],
                    C=C.transpose(),
                    nonlinearity=nonlinearity,
                    name='Sparse %d' % i
                )
                self.layers.append(sparse)
        else:
            for i, C in enumerate(diagram):
                sparse = SparseLayer(
                    self.layers[-1],
                    C=C.transpose(),
                    W=W[i].transpose(),
                    nonlinearity=nonlinearity,
                    name='Sparse %d' % i
                )
                self.layers.append(sparse)

        output_l = layers.DenseLayer(
            self.layers[-1],
            num_units=num_classes,
            nonlinearity=nonlinearities.softmax,
            name='Output'
        )
        self.layers.append(output_l)

        self.net = output_l

        self.predictions = layers.get_output(self.net)
        self.predict = theano.function([self.X_batch], self.predictions)
        self.pure_loss = T.mean(objectives.categorical_crossentropy(self.predictions, self.y_batch))

        self.regularization = sum([
                                      T.mean(param ** 2)
                                      for param in layers.get_all_params(self.net, regularizable=True)
                                      ]
                                  )

        self.regularization_coef = T.fscalar('regularization_coef')

        self.loss = self.pure_loss + self.regularization_coef * self.regularization

        self.learning_rate = T.fscalar('learning rate')
        params = layers.get_all_params(self.net)

        # upd = updates.adadelta(self.loss, params, learning_rate=self.learning_rate)
        upd = updates.momentum(self.loss, params, learning_rate=self.learning_rate)

        self.train = theano.function(
            [self.X_batch, self.y_batch, self.regularization_coef, self.learning_rate],
            self.pure_loss,
            updates=upd
        )

        self.get_loss = theano.function([self.X_batch, self.y_batch], self.pure_loss)

        super(DeepSparseNet, self).__init__()

    @staticmethod
    def batch_stream(n, batch_size=32):
        n_batches = int(n / batch_size)

        for i in range(n_batches):
            indx = np.random.choice(n, size=batch_size)
            yield indx

    def calculate_accuracy(self, X, y):
        y_pr = self.predict(X.astype(dtype='float32')).argmax(axis=1)

        # y_pr2 = np.zeros(shape=(y.shape[0], n_classes), dtype='float32')

        # onehot[np.arange(y.shape[0]), y] = 1.0

        num_correct = y[np.arange(y.shape[0]), y_pr].sum()

        # num_correct = np.logical_and(y, y_pr).all(axis=1).sum()
        return float(num_correct) / y.shape[0]

    def calculate_f_measure(self, X, y):
        y_pr = self.predict(X.astype(dtype='float32')).argmax(axis=1)

        f_total = 0.
        for j in range(y.shape[1]):
            real_positive = set(np.nonzero(y[:,j].squeeze())[0].tolist())
            predicted_positive = set(np.where(y_pr == j)[0].tolist())
            tp = float(len(real_positive & predicted_positive))
            precision = tp / (len(real_positive) + EPSILON)
            recall = tp / (len(predicted_positive) + EPSILON)
            f_total += 2*precision*recall/(precision+recall + EPSILON)

        return f_total / y.shape[1]

    def fit(self, X, y, n_epoches=1, batch_size=32, regularization_coef=1.0e-3, learning_rate=1.0):
        regularization_coef = np.float32(regularization_coef)
        learning_rate = np.float32(learning_rate)

        n_batches = int(X.shape[0] / batch_size)
        losses = np.zeros(shape=(n_epoches, n_batches), dtype='float32')

        for epoch in range(n_epoches):
            for i, indx in enumerate(self.batch_stream(X.shape[0], batch_size=batch_size)):
                X_norm = (X[indx] - X[indx].mean(axis=0)) / (X[indx].std(axis=0) + 0.0001)
                losses[epoch, i] = self.train(X_norm, y[indx], regularization_coef, learning_rate)
                # print(epoch, '\n', self.predict(X_norm))

            yield losses[epoch]
            #yield losses[:(epoch + 1)]

    def get_param_values(self):
        return layers.get_all_param_values(self.net)

    def set_param_values(self, param_values):
        layers.set_all_param_values(self.net, param_values)


class DeepDenseNet(Net):
    def __init__(self, diagram, num_classes, W=None, nonlinearity=nonlinearities.sigmoid):
        self.X_batch = T.fmatrix(name='X_batch')
        self.y_batch = T.fmatrix(name='y_batch')

        self.layers = []
        input_l = layers.InputLayer(shape=(None,) + (diagram[0].shape[1],), input_var=self.X_batch, name='Input')
        self.layers.append(input_l)

        reshape_l = layers.FlattenLayer(input_l, name='Reshape')
        self.layers.append(reshape_l)

        neurons = [20, 10]

        if W is None:
            for i, num_neurons in enumerate(neurons):
                sparse = layers.DenseLayer(
                    self.layers[-1],
                    num_units=num_neurons,
                    nonlinearity=nonlinearity,
                    name='Dense %d' % i
                )
                self.layers.append(sparse)
        else:
            for i, num_neurons in enumerate(neurons):
                sparse = layers.DenseLayer(
                    self.layers[-1],
                    num_units=num_neurons,
                    nonlinearity=nonlinearity,
                    name='Dense %d' % i
                )
                self.layers.append(sparse)

        output_l = layers.DenseLayer(
            self.layers[-1],
            num_units=num_classes,
            nonlinearity=nonlinearities.softmax,
            name='Output'
        )
        self.layers.append(output_l)

        self.net = output_l

        self.predictions = layers.get_output(self.net)
        self.predict = theano.function([self.X_batch], self.predictions)
        self.pure_loss = T.mean(objectives.categorical_crossentropy(self.predictions, self.y_batch))

        self.regularization = sum([
                                      T.mean(param ** 2)
                                      for param in layers.get_all_params(self.net, regularizable=True)
                                      ]
                                  )

        self.regularization_coef = T.fscalar('regularization_coef')

        self.loss = self.pure_loss + self.regularization_coef * self.regularization

        self.learning_rate = T.fscalar('learning rate')
        params = layers.get_all_params(self.net)

        # upd = updates.adadelta(self.loss, params, learning_rate=self.learning_rate)
        upd = updates.momentum(self.loss, params, learning_rate=self.learning_rate)

        self.train = theano.function(
            [self.X_batch, self.y_batch, self.regularization_coef, self.learning_rate],
            self.pure_loss,
            updates=upd
        )

        self.get_loss = theano.function([self.X_batch, self.y_batch], self.pure_loss)

        super(DeepDenseNet, self).__init__()

    @staticmethod
    def batch_stream(n, batch_size=32):
        n_batches = int(n / batch_size)

        for i in range(n_batches):
            indx = np.random.choice(n, size=batch_size)
            yield indx

    def calculate_accuracy(self, X, y):
        y_pr = self.predict(X.astype(dtype='float32')).argmax(axis=1)

        # y_pr2 = np.zeros(shape=(y.shape[0], n_classes), dtype='float32')

        # onehot[np.arange(y.shape[0]), y] = 1.0

        num_correct = y[np.arange(y.shape[0]), y_pr].sum()

        # num_correct = np.logical_and(y, y_pr).all(axis=1).sum()
        return float(num_correct) / y.shape[0]

    def calculate_f_measure(self, X, y):
        y_pr = self.predict(X.astype(dtype='float32')).argmax(axis=1)

        f_total = 0.
        for j in range(y.shape[1]):
            real_positive = set(np.nonzero(y[:,j].squeeze())[0].tolist())
            predicted_positive = set(np.where(y_pr == j)[0].tolist())
            tp = float(len(real_positive & predicted_positive))
            precision = tp / (len(real_positive) + EPSILON)
            recall = tp / (len(predicted_positive) + EPSILON)
            f_total += 2*precision*recall/(precision+recall + EPSILON)

        return f_total / y.shape[1]

    def fit(self, X, y, n_epoches=1, batch_size=32, regularization_coef=1.0e-3, learning_rate=1.0):
        regularization_coef = np.float32(regularization_coef)
        learning_rate = np.float32(learning_rate)

        n_batches = int(X.shape[0] / batch_size)
        losses = np.zeros(shape=(n_epoches, n_batches), dtype='float32')

        for epoch in range(n_epoches):
            for i, indx in enumerate(self.batch_stream(X.shape[0], batch_size=batch_size)):
                X_norm = (X[indx] - X[indx].mean(axis=0)) / (X[indx].std(axis=0) + 0.0001)
                losses[epoch, i] = self.train(X_norm, y[indx], regularization_coef, learning_rate)
                # print(epoch, '\n', self.predict(X_norm))

            yield losses[epoch]
            #yield losses[:(epoch + 1)]

    def get_param_values(self):
        return layers.get_all_param_values(self.net)

    def set_param_values(self, param_values):
        layers.set_all_param_values(self.net, param_values)
