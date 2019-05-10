import numpy as np
from activations import *
from optimizers import *
from functions import *

class Dense:
    def __init__(self, input_shape=(784, ), output_shape=(10, ), activation='Relu', optimizer='SGD', eps=0.01):
        self.W = np.sqrt(2.0 / input_shape[0]) * np.random.randn( input_shape[0], output_shape[0] )
        self.b = np.zeros(output_shape[0])
        self.x_shape = None
        self.x = None
        self.u = None
        self.activation = act[activation]()
        self.optimizer = opt[optimizer](eps=eps)

    def forward(self, x):
        self.x_shape = x.shape
        self.x = x.reshape(x.shape[0], -1)


        u = np.dot(self.x, self.W) + self.b
        self.u = u

        y = self.activation.forward(self.u)

        return y

    def backward(self, delta):
        dx = self.activation.backward(delta)

        self.dW = np.dot(self.x.T, dx)
        self.db = np.sum(dx, axis=0)

        dx = np.dot(dx, self.W.T)
        dx = dx.reshape(*self.x_shape)

        self.optimizer.update([self.W, self.b], [self.dW, self.db])

        return dx


class Conv:
    def __init__(self, kernels=8, input_shape=(1,28,28), conv_shape=(5,5), conv_pad=0, conv_stride=1,
                pool_shape=(2,2), pool_pad=0, pool_stride=2,
                activation='Relu', optimizer='SGD', eps=0.01):
        self.W = np.sqrt(2.0/ (conv_shape[0] * conv_shape[1])) * np.random.randn( kernels, input_shape[0], conv_shape[0], conv_shape[1] )
        self.b = np.zeros(kernels)
        self.W_col = None
        self.conv_pad = conv_pad
        self.conv_stride = conv_stride
        self.pool_shape = pool_shape
        self.pool_pad = pool_pad
        self.pool_stride = pool_stride
        self.x_shape = None
        self.x = None
        self.x_col = None
        self.u = None
        self.u_col = None
        self.conv_y = None
        self.conv_y_col = None
        self.activation = act[activation]()
        self.optimizer = opt[optimizer](eps=eps)

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = 1 + int((H + 2*self.conv_pad - FH) / self.conv_stride)
        out_w = 1 + int((W + 2*self.conv_pad - FW) / self.conv_stride)

        col = im2col(x, FH, FW, self.conv_stride, self.conv_pad)
        col_W = self.W.reshape(FN, -1).T

        self.u_col = np.dot(col, col_W) + self.b
        self.u = self.u_col.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        self.x = x
        self.x_col = col
        self.W_col = col_W


        self.conv_y = self.activation.forward(self.u)
        y = self.conv_y

        if self.pool_shape[0] != 0 and self.pool_shape[1] != 0:
            N, C, H, W = self.conv_y.shape
            out_h = 1 + int((H + 2*self.pool_pad - self.pool_shape[0]) / self.pool_stride)
            out_w = 1 + int((W + 2*self.pool_pad - self.pool_shape[1]) / self.pool_stride)

            self.conv_y_col = im2col(self.conv_y, self.pool_shape[0], self.pool_shape[1], self.pool_stride, self.pool_pad)
            self.conv_y_col = self.conv_y_col.reshape(-1, self.pool_shape[0]*self.pool_shape[1])

            arg_max = np.argmax(self.conv_y_col, axis=1)
            self.pool_y = np.max(self.conv_y_col, axis=1)
            self.pool_y = self.pool_y.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)
            y = self.pool_y

            self.arg_max = arg_max

        return y

    def backward(self, delta):
        if self.pool_shape[0] != 0 and self.pool_shape[1] != 0:
            delta = delta.transpose(0, 2, 3, 1)
            pool_size = self.pool_shape[0] * self.pool_shape[1]
            dmax = np.zeros((delta.size, pool_size))
            dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = delta.flatten()
            dmax = dmax.reshape(delta.shape + (pool_size,))

            dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
            delta = col2im(dcol, self.conv_y.shape, self.pool_shape[0], self.pool_shape[1], self.pool_stride, self.pool_pad)


        FN, C, FH, FW = self.W.shape

        dx = self.activation.backward(delta)
        dx = dx.transpose(0,2,3,1).reshape(-1, FN)

        self.db = np.sum(dx, axis=0)
        self.dW = np.dot(self.x_col.T, dx)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        dx = np.dot(dx, self.W_col.T)
        dx = col2im(dx, self.x.shape, FH, FW, self.conv_stride, self.conv_pad)

        self.optimizer.update([self.W, self.b], [self.dW, self.db])

        return dx
