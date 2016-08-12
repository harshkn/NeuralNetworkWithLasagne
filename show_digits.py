from __future__ import print_function

import sys
import os
import time

import theano
from theano import tensor as T
import numpy as np
from load_data import load_dataset
import lasagne
import matplotlib.pyplot as plt

# print("Loading data...")
# X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
# print(X_train.shape)
# print(y_train.shape)
# print(X_val.shape)
# print(y_val.shape)
# print(X_test.shape)
# print(y_test.shape)



print("Loading data...")
X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
plt.figure(figsize=(12,3))
for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.imshow(X_train[i].reshape((28, 28)), cmap='gray', interpolation='nearest')
    plt.axis('off')
    plt.savefig('digits.png',bbox_inches='tight')

