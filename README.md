# Deep Learning Library (Implemented with Numpy)

## Requirements
1. Python 2/3
2. Numpy
3. Scikit-Learn

## Usage
1. To create a neural network with 2 layers and softmax at the end,
```python
l1 = hLayer(784, 65, ac=relu, batchN=False, l2=0.0, l1=0.0)
# ac can be relu or sigmoid
# batchN refers to batchnormalization enabled or disabled
# l2 and l1 are regularization paramteres
l2 = hLayer(65, 9, batchN=False)
l3 = softmax_layer(9, 9)
m = Model([l1, l2, l3])
```
2.  To train the model using adam optimizer and cross entropy loss,

```python
m.backprop_cross_multi(X, Y, alpha=0.0002, opti='adam')
```

3. For evaluation

```python
out = m.forward_pass(X)
```

Note: `XOR_perceptron.py` and `XOR_hidden.py` are written in keras for demonstration purposes of XOR problem. Run directly to get the decision boundary for the corresponding models.
