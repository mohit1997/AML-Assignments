import numpy as np
import math

def mini_batches(X, Y, mini_batch_size):

    np.random.seed(0)
    m = X.shape[1]
    mini_batches = []

    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation]

    num_complete_minibatches = math.floor(m/mini_batch_size)

    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, mini_batch_size*k:mini_batch_size*(k+1)]
        mini_batch_Y = shuffled_Y[:, mini_batch_size*k:mini_batch_size*(k+1)]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches*mini_batch_size:m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches*mini_batch_size:m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


def sigmoid(inp, ret=False):
	out = 1 /(1+np.exp(-inp))
	if ret:
		return out, np.multiply(out,(1-out))
	else:
		return out

def relu(inp, ret=False):
	x = np.maximum(inp, 0)
	# x = abs(inp)/2 + inp/2

	grad = x /(inp + 1e-8)

	# print(x)
	# temp = np.count_nonzero(grad)
	# print("None Zero", temp/float(x.shape[0]*x.shape[1]))

	if ret:
		return x, grad
	else:
		return x

def softmax(inputs):
	out = inputs - np.max(inputs, axis=1).reshape(-1, 1)
	temp = np.exp(out)
	sumn = np.sum(temp, axis=1).reshape(-1, 1)
	fin = temp/sumn
	# for i in fin:
	# 	for j in i:
	# 		if j>1 or j<0:
	# 			print("Error")
	return fin

def get_grad_soft(k, out, delta):
		M = k
		for i in range(M):
			temp = out
			temp1 = np.multiply(temp, temp[:, i].reshape(-1, 1))
			temp1 = np.multiply(np.multiply(temp[:, i].reshape(-1, 1), temp), delta)
			grad[:, i] = np.sum(temp1, axis=1)
		return grad

class Model:

	layers = []

	def __init__(self, l):
		if type(l) is not list:
			print("Error! The model will not work")
		else:
			self.layers = l




	def create_layer(self, inp, out):
		self.layers.append(Layer(inp, out))

	def forward_pass(self, inputs):

		# print("FORWARD___________________")
		for layer in self.layers:
			out_temp = layer.forward_pass(inputs)
			# print(out_temp, layer.name)
			inputs = out_temp
		# print("FORWARD___________________")
		return inputs

	def test_mode(self, inputs):
		for layer in self.layers:
			out_temp = layer.forward_pass(inputs, mode="test")
			# print(out_temp, layer.name)
			inputs = out_temp
		# print("FORWARD___________________")
		return inputs

	def cross_loss(self, out, lab):
		temp = -np.mean(np.sum(np.multiply(np.log(out + 1e-8), lab), axis=1), axis=0)
		return temp

	def mean_squared_loss(self, out, lab):
		temp = out - lab
		return np.mean(temp**2)

	def accuracy(self, pred, labels):
		pred_val = np.argmax(pred, axis=1)
		lab_val = np.argmax(labels, axis=1)
		c = pred_val == lab_val
		print("Accuracy", np.sum(c, dtype=float)/float(len(c)))
		# print("Accuracy", np.sum(c)/len(c))import numpy as np

	def backprop_cross_multi(self, inputs, labels, alpha=0.01, opti='adam'):
		delta = (self.forward_pass(inputs) - labels)/float(len(labels))

		for layer in reversed(self.layers):
			if layer.name == "softmax":
				delta = layer.backprop(delta, alpha, True)
			else:
				delta = layer.backprop(delta, alpha, opti=opti)

	def backprop_cross_bin(self, inputs, labels, alpha=0.01, opti='adam'):

		pred = self.forward_pass(inputs)
		temp = np.multiply(pred, 1 - pred)/float(len(labels))
		delta = (pred - labels)/temp
		i=0
		# print(delta + labels)
		for layer in reversed(self.layers):
			delta = layer.backprop(delta, alpha, opti='adam')



class Layer(object):

	def forward_pass(self, inputs):

		return []

	def backprop(self, deltain):

		return []


class hLayer(Layer):

	def __init__(self, in_sz, out_sz, ac=None, batchN=False, l1=0, l2=0):
		self.name = "Hidden Layer"
		self.insize = in_sz
		self.outsize = out_sz
		if ac == sigmoid:
			self.W = np.random.normal(0, np.sqrt(1/float(in_sz)), (in_sz, out_sz))
		elif ac == relu:
			self.W = np.random.normal(0, np.sqrt(2/float(in_sz)), (in_sz, out_sz))
		else:
			self.W = np.random.randn(in_sz, out_sz)*0.01
		# self.W = np.random.normal(0, np.sqrt(2)/float(in_sz), (in_sz, out_sz))
		self.Wconfig = None #adam
		self.b = np.zeros((1, out_sz), dtype=float)
		self.bconfig = None #adam
		self.bn = batchN
		self.l1 = l1
		self.l2 = l2
		if self.bn:
			self.epsilon = 1e-8
			self.gamma = np.random.normal(0, 1/float(out_sz), (1, out_sz))
			self.beta = np.zeros((1, out_sz))
			self.mu = 0
			self.var = 0
			self.gammaconfig = None #adam
			self.betaconfig = None #adam
		self.activation = ac

	def bn_back_prop(self, deltain, v=False):
		# print('Hello')
		N, M = deltain.shape
		x_mu = self.x_mu
		inv = 1/self.std_dev

		dx_cap = deltain * self.gamma
		dvar = np.sum(dx_cap * x_mu, axis=0)*-0.5*inv**3
		dmu = np.sum(dx_cap * -inv, axis=0) + dvar*np.mean(-2*x_mu, axis=0)

		dX = (dx_cap*inv) + dvar*2*x_mu/float(N) + dmu/float(N)

		dgamma = np.sum(deltain*self.x_cap, axis=0)
		dbeta = np.sum(deltain, axis=0)

		return dX, dgamma, dbeta

	def forward_pass(self, inputs, mode="train"):

		if mode=="test":
			self.inp = inputs
			self.out = inputs.dot(self.W) + self.b
			self.grad = 1
			if self.bn:
				mu = np.mean(self.out, axis=0, keepdims=True)
				var = np.var(self.out, axis=0, keepdims=True)
				x_mu = self.out - mu
				x_cap = x_mu/np.sqrt((var + self.epsilon))
				self.out = np.multiply(self.gamma,x_cap) + self.beta
				# print(out.shape)
			if self.activation is not None:
				self.out, self.grad = self.activation(self.out, ret=True)
			return self.out

		self.inp = inputs
		self.out = inputs.dot(self.W) + self.b
		self.grad = 1
		if self.bn:
			self.mu = 0.9*self.mu + 0.1*np.mean(self.out, axis=0, keepdims=True)
			self.var = 0.9*self.var + 0.1*np.var(self.out, axis=0, keepdims=True)
			# self.mu = np.mean(self.out, axis=0, keepdims=True)
			# self.var = np.var(self.out, axis=0, keepdims=True)
			x_mu = self.out - self.mu
			x_cap = x_mu/np.sqrt((self.var + self.epsilon))
			self.x_cap = x_cap
			self.x_mu = x_mu
			self.std_dev = np.sqrt(self.var + self.epsilon)
			self.out = np.multiply(self.gamma,x_cap) + self.beta
			# print(out.shape)
		if self.activation is not None:
			self.out, self.grad = self.activation(self.out, ret=True)
		return self.out

	def backprop(self, deltain, alpha=0.001, opti='adam'):
		X = self.inp
		# print("deltain", deltain.shape)
		# print("Weights", self.W)
		delta = deltain
		if self.activation is not None:
			delta = np.multiply(self.grad, delta)
		if self.bn:
			delta, dgamma, dbeta = self.bn_back_prop(delta, False)
			if opti == 'sgd':
				self.gamma = self.gamma - alpha*dgamma
				self.beta = self.beta - alpha*dbeta
			elif opti == "adam":
				# print('adam')
				self.gamma, self.gammaconfig = self.adam(self.gamma, dgamma, config=self.gammaconfig, alpha=alpha)
				self.beta, self.betaconfig = self.adam(self.gamma, dgamma, config=self.betaconfig, alpha=alpha)
			# print("batchnormalized layer", dbeta)
		N, M = delta.shape
		deltaw = X.T.dot(delta) + self.l2*0.5*self.W/float(N) + self.l1*np.sign(self.W)/float(N)
		deltab = np.sum(delta, axis=0, keepdims=True)
		# print("Gradient", deltaw)
		if opti == "sgd":
			self.W = self.W - alpha*deltaw
			self.b = self.b - alpha*deltab
		elif opti == "adam":
			# print('adam')
			self.W, self.Wconfig = self.adam(self.W, deltaw, config=self.Wconfig, alpha=alpha)
			self.b, self.bconfig = self.adam(self.b, deltab, config=self.bconfig, alpha=alpha)

		return delta.dot(self.W.T)

	def adam(self, x, dx, config=None, alpha=0.001):
		"""
		Uses the Adam update rule, which incorporates moving averages of both the
		gradient and its square and a bias correction term.
		config format:
		- learning_rate: Scalar learning rate.
		- beta1: Decay rate for moving average of first moment of gradient.
		- beta2: Decay rate for moving average of second moment of gradient.
		- epsilon: Small scalar used for smoothing to avoid dividing by zero.
		- m: Moving average of gradient.
		- v: Moving average of squared gradient.
		- t: Iteration number.
		"""
		if config is None: config = {}
		config.setdefault('learning_rate', alpha)
		config.setdefault('beta1', 0.9)
		config.setdefault('beta2', 0.999)
		config.setdefault('epsilon', 1e-8)
		config.setdefault('m', np.zeros_like(x))
		config.setdefault('v', np.zeros_like(x))
		config.setdefault('t', 0)

		next_x = None
		beta1, beta2, eps = config['beta1'], config['beta2'], config['epsilon']
		t, m, v = config['t'], config['m'], config['v']
		m = beta1 * m + (1 - beta1) * dx
		v = beta2 * v + (1 - beta2) * (dx * dx)
		t += 1
		alpha = config['learning_rate'] * np.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
		x -= alpha * (m / (np.sqrt(v) + eps))
		config['t'] = t
		config['m'] = m
		config['v'] = v
		next_x = x
		# print(next_x)

		return next_x, config

class softmax_layer(Layer):

	def __init__(self, in_sz, out_sz):
		self.name = "softmax"
		self.insize = in_sz
		self.outsize = out_sz
		if(in_sz != out_sz):
			print("Error Implementing Softmax, Check Dimensions")


	def forward_pass(self, inputs, mode="train"):
		self.inp = inputs
		self.out = softmax(inputs)
		return self.out


	def backprop(self, deltain, alpha=0.001, cross=False):
		if cross:
			return deltain
		grad = get_grad_soft(self.insize, self.out, deltain)
		return grad
