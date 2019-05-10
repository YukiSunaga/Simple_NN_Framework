import numpy as np
from layers import *
from functions import *
import sys


class Model:
    def __init__(self, epochs=20, batch_size=32):
        self.layers = []
        self.epochs = epochs
        self.batch_size = batch_size

    def predict(self, x, train_flg=False):
        for i, layer in enumerate(self.layers):
            x = layer.forward(x, train_flg)
        return x

    def classify(self, x, train_flg=False):
        y = self.predict(x, train_flg)
        y = np.argmax(y ,axis=1)

        return y

    def loss(self, x, t, train_flg=False):
        y = self.predict(x, train_flg)
        y = cross_entropy(y, t)
        return y

    def accuracy(self, x, t):
        y = self.classify(x, train_flg=False)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        acc = np.sum(y == t) / float(x.shape[0])
        return acc

    def loss_acc(self, x, t):
        y = self.predict(x, train_flg=False)
        loss = cross_entropy(y, t)

        y = np.argmax(y ,axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        acc = np.sum(y == t) / float(x.shape[0])
        return loss, acc

    def _fit(self, x, t):
        loss = self.loss(x, t, train_flg=True)
        layers_r = self.layers.copy()
        layers_r.reverse()
        for i, layer in enumerate(layers_r):
            t = layer.backward(t)
        return loss

    def fit(self, x_train, y_train, x_test, y_test):
        iter_per_epoch = int(max(x_train.shape[0] / self.batch_size, 1))
        max_iter = int(self.epochs * iter_per_epoch)
        iter = 0
        piter = 0
        epoch = 0
        loss = 0

        for i in range(max_iter):
            batch_mask = np.random.choice(x_train.shape[0], self.batch_size)
            x_batch = x_train[batch_mask]
            y_batch = y_train[batch_mask]

            loss += self._fit(x_batch, y_batch)

            sys.stdout.write("\repoch %d | %d / %d | train_loss %f" %(epoch, piter, iter_per_epoch, loss/(piter+1)))
            sys.stdout.flush()

            if iter % iter_per_epoch == 0:
                epoch += 1

                size = int(x_test.shape[0]/32)
                t_loss = 0
                t_acc = 0
                for i in range(size):
                    loss, acc = self.loss_acc(x_test[32*i:32*(i+1)], y_test[32*i:32*(i+1)])
                    t_loss += loss
                    t_acc += acc
                t_loss /= size
                t_acc /= size
                print('\n ', 'test_loss', t_loss, 'test_acc', t_acc)
                piter = 0
                loss = 0

            iter += 1
            piter += 1


        size = int(x_train.shape[0]/32)
        tr_loss = 0
        tr_acc = 0
        for i in range(size):
            loss, acc = self.loss_acc(x_train[32*i:32*(i+1)], y_train[32*i:32*(i+1)])
            tr_loss += loss
            tr_acc += acc
        tr_loss /= size
        tr_acc /= size


        size = int(x_test.shape[0]/32)
        ts_loss = 0
        ts_acc = 0
        for i in range(size):
            loss, acc = self.loss_acc(x_test[32*i:32*(i+1)], y_test[32*i:32*(i+1)])
            ts_loss += loss
            ts_acc += acc
        ts_loss /= size
        ts_acc /= size

        print('\n\n', 'train_loss', tr_loss, 'train_acc', tr_acc, 'test_loss', ts_loss, 'test_acc', ts_acc)


class Basic_Model(Model):
    def __init__(self, epochs=20, batch_size=32):
        super().__init__(epochs, batch_size)
        self.layers.append(Dense(activation='Softmax'))

class Hid_Model(Model):
    def __init__(self, epochs=20, batch_size=32, hid_size=128):
        super().__init__(epochs, batch_size)
        self.layers.append(Dense(input_shape=(784, ), output_shape=(hid_size, ), activation='Relu'))
        self.layers.append(Dense(input_shape=(hid_size, ), output_shape=(10, ), activation='Softmax'))

class Conv_Model(Model):
    def __init__(self, epochs=20, batch_size=32):
        super().__init__(epochs, batch_size)
        self.layers.append(Conv(kernels=16, input_shape=(1,28,28), conv_shape=(5,5), conv_pad=0, pool_shape=(2,2), activation='Tanh', optimizer='Adam', eps=0.001))
        self.layers.append(Dense(input_shape=(16*12*12, ), output_shape=(64, ), activation='Tanh', optimizer='Adam', eps=0.001))
        self.layers.append(Dense(input_shape=(64, ), output_shape=(10, ), activation='Softmax', optimizer='Adam', eps=0.001))


class Conv_Model_DOBN(Model):
    def __init__(self, epochs=20, batch_size=32, kernels=32, hid_size=128, batchnorm=True, dropout=True):
        super().__init__(epochs, batch_size)
        self.layers.append(Conv(kernels=kernels, input_shape=(1,28,28), conv_shape=(5,5), conv_pad=0, pool_shape=(2,2), activation='Relu', dropout=dropout, batchnorm=batchnorm, optimizer='Adam', eps=0.001))
        self.layers.append(Dense(input_shape=(kernels*12*12, ), output_shape=(hid_size, ), activation='Relu', dropout=dropout, batchnorm=batchnorm, optimizer='Adam', eps=0.001))
        self.layers.append(Dense(input_shape=(hid_size, ), output_shape=(10, ), activation='Softmax', optimizer='Adam', batchnorm=batchnorm, eps=0.001))



class Conv_Model2(Model):
    def __init__(self, epochs=20, batch_size=32):
        super().__init__(epochs, batch_size)
        self.layers.append(Conv(kernels=32, input_shape=(1,28,28), conv_shape=(5,5), conv_pad=2, pool_shape=(0,0), optimizer='Adam', eps=0.001))
        self.layers.append(Conv(kernels=32, input_shape=(32,28,28), conv_shape=(5,5), conv_pad=2, pool_shape=(2,2), optimizer='Adam', eps=0.001))
        self.layers.append(Conv(kernels=32, input_shape=(32,14,14), conv_shape=(5,5), conv_pad=2, pool_shape=(2,2), optimizer='Adam', eps=0.001))
        self.layers.append(Dense(input_shape=(32*7*7, ), output_shape=(128, ), activation='Relu', optimizer='Adam', eps=0.001))
        self.layers.append(Dense(input_shape=(128, ), output_shape=(10, ), activation='Softmax', optimizer='Adam', eps=0.001))
