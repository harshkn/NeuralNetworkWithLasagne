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


with np.load('model_cnn.npz') as f:
     param_values = [f['arr_%d' % i] for i in range(len(f.files))]

# for idx,param in enumerate(param_values):
# 	print(idx)
# 	print(param.squeeze().shape)
# 	sparam = param.squeeze()
# 	sparam = sparam.reshape((32, 25))
# 	plt.imshow(sparam)

print(param_values[0].shape)
sparam = param_values[0].squeeze()
sparam = sparam.reshape((32, 25))
plt.imshow(sparam, cmap = plt.get_cmap('gray'))
# plt.title('Training and Validation loss')
# plt.legend(loc='upper right')
plt.savefig('layer_1.png',bbox_inches='tight')


# sparam = param_values[2].squeeze()
# sparam = sparam.reshape((32*32, 25))
# plt.imshow(sparam, cmap = plt.get_cmap('gray'))
# # plt.title('Training and Validation loss')
# # plt.legend(loc='upper right')
# plt.savefig('layer_2.png',bbox_inches='tight')


# sparam = param_values[4].squeeze()
# # sparam = sparam.reshape((32*32, 25))
# plt.imshow(sparam, cmap = plt.get_cmap('gray'))
# # plt.title('Training and Validation loss')
# plt.legend(loc='upper right')
# plt.savefig('layer_3.png',bbox_inches='tight')



# sparam = param_values[6].squeeze()
# # sparam = sparam.reshape((32*32, 25))
# plt.imshow(sparam, cmap = plt.get_cmap('gray'))
# # plt.title('Training and Validation loss')
# # plt.legend(loc='upper right')
# plt.savefig('layer_4.png',bbox_inches='tight')



