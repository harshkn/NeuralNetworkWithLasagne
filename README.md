# TwoLayerNNWithLasagne
Simple two layer neural network with Theano and Lasagne to do handwritten digit recognition.

####Sample Digits
![Sample digits](digits.png)

####Model 
Input Layer size: 784 with 20% drop-out and ReLU activation   
Hidden Layer size: 784 x 625 with 50% drop-out with ReLU activation  
Output Layer: 625 x 10 with 50% drop-out with softmax activation  

####Without using mini-batch: epoch - 500, test accuracy 93%
Tried to train a simple 2 layer network with no minibatch. Got around 93% test accuracy after 500 epochs. 
######Training loss and Validation loss without using mini-batch

![Training loss Vs Validation loss](loss_no_mb.png)  
<img src="loss_no_mb.png" alt="alt text" width="100" height="80">


####With using mini-batch: epoch - 150, test accuracy 98.15%
With using mini-batch the learning is faster.
######Training loss and Validation loss with mini-batch
![Training loss Vs Validation loss](loss_mb.png)





