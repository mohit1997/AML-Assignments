import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from re_ass4 import *

import os
import struct

import warnings
warnings.filterwarnings("ignore")

fname_img = os.path.join('emnist-balanced-train-images-idx3-ubyte')
fname_lbl = os.path.join('emnist-balanced-train-labels-idx1-ubyte')

with open(fname_lbl, 'rb') as flbl:
    magic, num = struct.unpack(">II", flbl.read(8))
    lbl = np.fromfile(flbl, dtype=np.int8)

with open(fname_img, 'rb') as fimg:
    magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)
    img = np.transpose(img, (0, 2, 1))
    img = img.reshape(img.shape[0], -1)

get_img = lambda idx: (lbl[idx], img[idx])

from sklearn.preprocessing import LabelBinarizer

req_ind = (lbl == 10) | (lbl == 11) | (lbl == 17) | (lbl == 21) | (lbl == 23) | (lbl == 24) | (lbl == 27) | (lbl == 31) | (lbl == 34)
train_labels = lbl[req_ind]

lb = LabelBinarizer()
lb.fit([10, 11, 17, 21, 23, 24, 27, 31, 34])

y_train = pd.DataFrame(lb.transform(train_labels))/1.0
X_train = pd.DataFrame(img[req_ind])

X = np.array(X_train)
# print(X_train.shape)
# print(y_train.shape)


fname_img = os.path.join('emnist-balanced-test-images-idx3-ubyte')
fname_lbl = os.path.join('emnist-balanced-test-labels-idx1-ubyte')

with open(fname_lbl, 'rb') as flbl:
    magic, num = struct.unpack(">II", flbl.read(8))
    lbl = np.fromfile(flbl, dtype=np.int8)

with open(fname_img, 'rb') as fimg:
    magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)
    img = np.transpose(img, (0, 2, 1))
    img = img.reshape(img.shape[0], -1)

get_img = lambda idx: (lbl[idx], img[idx])

req_ind = (lbl == 10) | (lbl == 11) | (lbl == 17) | (lbl == 21) | (lbl == 23) | (lbl == 24) | (lbl == 27) | (lbl == 31) | (lbl == 34)
train_labels = lbl[req_ind]

lb = LabelBinarizer()
lb.fit([10, 11, 17, 21, 23, 24, 27, 31, 34])

y_test = pd.DataFrame(lb.transform(train_labels))/1.0
X_test = pd.DataFrame(img[req_ind])

# print(X_test.shape)
# print(y_test.shape)

l1 = hLayer(784, 65, ac=relu, batchN=False, l2=0.0, l1=0.0)
# l2 = hLayer(50, 50, ac=relu, batchN=True, l2=0.0, l1=0.0)
l4 = hLayer(65, 9, batchN=False)
l5 = softmax_layer(9, 9)

m = Model([l1, l4, l5])

X_test = np.array(X_test)/255.0
Y_T = np.array(y_test)

# def one_hot(inputs, k):
# 	out = np.zeros((len(inputs), k), dtype=float)
# 	out[np.arange(len(inputs)), inputs.reshape(-1)] = 1
# 	# print(out)
# 	return out

# Y_ = one_hot(np.array(y_train), k=9)
X = np.array(X_train)/255.0
Y_ = np.array(y_train)
# print(Y_.shape)

batch_size = 128  # Approximately 25 samples per batch
nb_of_batches = X.shape[0] / batch_size  # Number of batches
# Create batches (X,Y) from the training set
XT_batches = list(np.array_split(X, nb_of_batches, axis=0))
YT_batches = list(np.array_split(Y_, nb_of_batches, axis=0))  # Y targets

max_nb_of_iterations = 200
counter=0
for iteration in range(max_nb_of_iterations):
    # counter += 1
    XT_batches = mini_batches(X.T, Y_.T, 128)
    for i in range(len(XT_batches)):
        X_in = XT_batches[i][0].T
        T_in = XT_batches[i][1].T
        # print('Hello')
        m.backprop_cross_multi(X_in, T_in, alpha=0.0002, opti='adam')
        # print("Count", i)
        # counter += 1
    counter+=1
    if(counter % 10 != 0):
    	continue

    print("Counter", counter)
    out1 = m.forward_pass(X_in)
    print(np.max(out1[0:10], axis=1))
    # print(out1.shape)
    l = m.cross_loss(out1, T_in)
    m.accuracy(out1, T_in)
    print(l)
    out1 = m.test_mode(X_test)
    # print(np.max(out1[0:10], axis=1))
    # print(out1.shape)
    print("Testing Accuracy is")
    l = m.cross_loss(out1, Y_T)
    m.accuracy(out1, Y_T)
    print(l)

out1 = m.forward_pass(X)
print(np.max(out1[0:10], axis=1))
# print(out1.shape)
l = m.cross_loss(out1, Y_)
m.accuracy(out1, Y_)
print(l)
