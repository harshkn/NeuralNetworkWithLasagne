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

def two_layer_model(input_var = None):
	l_in = lasagne.layers.InputLayer(shape=(None, 784),input_var=input_var)
	l_in_drop = lasagne.layers.DropoutLayer(l_in, p=0.2)
	l_hid1 = lasagne.layers.DenseLayer(l_in_drop, num_units=625,nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.GlorotUniform())
	l_hid1_drop = lasagne.layers.DropoutLayer(l_hid1, p=0.5)
	l_out = lasagne.layers.DenseLayer(l_hid1_drop, num_units=10,nonlinearity=lasagne.nonlinearities.softmax)
	return l_out

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

# def main(model='mlp', num_epochs=500):
    # Load the dataset
print("Loading data...")
X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
input_var = T.fmatrix('inputs')
target_var = T.ivector('targets')
print("Building model and compiling functions...")
network = two_layer_model(input_var)

prediction = lasagne.layers.get_output(network)
loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
loss = loss.mean()

params = lasagne.layers.get_all_params(network, trainable=True)
updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.01, momentum=0.9)

test_prediction = lasagne.layers.get_output(network, deterministic=True)
test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,target_var)

test_loss = test_loss.mean()

test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

train_fn = theano.function([input_var, target_var], loss, updates=updates)

val_fn = theano.function([input_var, target_var], [test_loss, test_acc])


# print("Starting training...")
# num_epochs = 500
# tr_loss = []
# val_loss = []

# for epoch in range(num_epochs):
#     # In each epoch, we do a full pass over the training data:
#     train_err = 0
#     start_time = time.time()
#     train_err += train_fn(X_train, y_train)

#     val_err = 0
#     val_acc = 0

#     err, acc = val_fn(X_val, y_val)
#     val_err += err
#     val_acc += acc
#     tr_loss.append(train_err)
#     val_loss.append(err)
    


#     print("Epoch {} of {} took {:.3f}s".format(
#         epoch + 1, num_epochs, time.time() - start_time))
#     print("  training loss:\t\t{:.6f}".format(train_err ))
#     print("  validation loss:\t\t{:.6f}".format(val_err ))
#     print("  validation accuracy:\t\t{:.2f} %".format(
#         val_acc  * 100))

print("Starting training...")
num_epochs = 150
tr_loss = []
val_loss = []

    # We iterate over epochs:
for epoch in range(num_epochs):
    # In each epoch, we do a full pass over the training data:
    train_err = 0
    train_batches = 0
    start_time = time.time()
    for batch in iterate_minibatches(X_train, y_train, 500, shuffle=True):
        inputs, targets = batch
        train_err += train_fn(inputs, targets)
        train_batches += 1

    # And a full pass over the validation data:
    val_err = 0
    val_acc = 0
    val_batches = 0
    for batch in iterate_minibatches(X_val, y_val, 500, shuffle=False):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        val_err += err
        val_acc += acc
        val_batches += 1

    tr_loss.append(train_err / train_batches)
    val_loss.append(val_err / val_batches)

    # Then we print the results for this epoch:
    print("Epoch {} of {} took {:.3f}s".format(
        epoch + 1, num_epochs, time.time() - start_time))
    print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
    print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
    print("  validation accuracy:\t\t{:.2f} %".format(
        val_acc / val_batches * 100))


test_err = 0
test_acc = 0
test_batches = 0
for batch in iterate_minibatches(X_test, y_test, 500, shuffle=False):
    inputs, targets = batch
    err, acc = val_fn(inputs, targets)
    test_err += err
    test_acc += acc
    test_batches += 1
print("Final results:")
print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
print("  test accuracy:\t\t{:.2f} %".format(
    test_acc / test_batches * 100))

plt.plot(range(num_epochs),tr_loss,label='Training Loss')
plt.plot(range(num_epochs),val_loss,label='Validation Loss')
# plt.axis('off')
fn = "loss_mb.png"
plt.xlabel('epoch')
plt.ylabel('Loss')
plt.title('Training and Validation loss')
plt.legend(loc='upper right')
plt.savefig(fn,bbox_inches='tight')


np.savez('model.npz', *lasagne.layers.get_all_param_values(network))

# with np.load('model.npz') as f:
#      param_values = [f['arr_%d' % i] for i in range(len(f.files))]
# lasagne.layers.set_all_param_values(network_output, param_values)




# if __name__ == '__main__':
# 	kwargs = {}
#     if len(sys.argv) > 1:
#         kwargs['model'] = sys.argv[1]
#     if len(sys.argv) > 2:
#         kwargs['num_epochs'] = int(sys.argv[2])
#     main(**kwargs)kwargs = {}
#     if len(sys.argv) > 1:
#         kwargs['model'] = sys.argv[1]
#     if len(sys.argv) > 2:
#         kwargs['num_epochs'] = int(sys.argv[2])
#     main(**kwargs)






