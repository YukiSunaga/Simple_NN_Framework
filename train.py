import numpy as np
from models import Basic_Model, Hid_Model, Conv_Model
from functions import to_one_hot_label
from fashion_mnist import load_fashion_mnist

(x_train, y_train), (x_test, y_test) = load_fashion_mnist(normalize=True, flatten=False, one_hot_label=True)
model = Conv_Model()

model.fit(x_train, y_train, x_test, y_test)
