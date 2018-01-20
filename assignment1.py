import numpy as np 
np.random.seed(0)
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
		for layer in self.layers:
			out_temp = layer.forward_(inputs)
			inputs = out_temp

		return inputs

	def mean_squared_loss(self, out, lab):
		temp = out - lab
		return np.mean(temp**2)

	def backprop_mse(self, inputs, lab, alpha = 0.001):
		delta_fin = self.forward_pass(inputs) - lab
		# print("delta_fin", delta_fin.shape, lab.shape, self.forward_pass(inputs).shape)
		# layer = self.layers[-1]
		

		for count, layer in enumerate(list(reversed(self.layers))):
			# print("This is Layer", len(self.layers) - count)
			# print(layer.in_size, layer.out_size)
			if layer.activation_f != None:
				grad = layer.grad
				delta = np.multiply(grad, delta_fin)
			else:
				delta = delta_fin
			# print("bias term", layer.bias.shape)
			layer.bias = layer.bias - alpha*delta
			# print("bias term", layer.bias.shape)
			if count != len(self.layers) - 1:
				delta_w = np.matmul((self.layers[count - 1].out).transpose(), delta)
				# print("Delta Shape", delta.shape)
				# print("Delta Weights Shape", delta_w.shape)
				# print(delta_w.shape)
				# print(count)
				# print("____________________")
				# print(layer.weight.shape)
				layer.weight = layer.weight - alpha*delta_w
				# print(layer.weight.shape)
				# print("____________________")
				delta_fin = np.matmul(delta, (layer.weight).transpose())
			else:
				delta_w = np.matmul((inputs).transpose(), delta)
				# print(delta_w.shape)
				# print(count)
				layer.weight = layer.weight - alpha*delta_w

	def backprop_bin_cross_entropy(self, inputs, lab, alpha = 0.001):
		delta_fin = self.forward_pass(inputs) - lab
		# print("delta_fin", delta_fin.shape, lab.shape, self.forward_pass(inputs).shape)
		# layer = self.layers[-1]
		

		for count, layer in enumerate(list(reversed(self.layers))):
			# print("This is Layer", len(self.layers) - count)
			# print(layer.in_size, layer.out_size)
			if layer.activation_f != None:
				if count != 0:
					grad = layer.grad
					delta = np.multiply(grad, delta_fin)
				else :
					delta = delta_fin
			else:
				if count == 0:
					print("Use sigmoid/softmax at output")
					break;
				delta = delta_fin
			# print("bias term", layer.bias.shape)
			layer.bias = layer.bias - alpha*delta
			# print("bias term", layer.bias.shape)
			if count != len(self.layers) - 1:
				delta_w = np.matmul((self.layers[count - 1].out).transpose(), delta)
				# print("Delta Shape", delta.shape)
				# print("Delta Weights Shape", delta_w.shape)
				# print(delta_w.shape)
				# print(count)
				# print("____________________")
				# print(layer.weight.shape)
				layer.weight = layer.weight - alpha*delta_w
				# print(layer.weight.shape)
				# print("____________________")
				delta_fin = np.matmul(delta, (layer.weight).transpose())
			else:
				delta_w = np.matmul((inputs).transpose(), delta)
				# print(delta_w.shape)
				# print(count)
				layer.weight = layer.weight - alpha*delta_w





def sigmoid(inputs, ret=False):
	a = lambda x : 1/(1+np.exp(-x))
	b = np.vectorize(a)
	temp = b(inputs)
	if ret:
		return temp, np.multiply(temp, 1-temp)
	else:
		return b(inputs)

def relu(inputs, ret=False):
	a = lambda x : x if x>=0 else 0
	b = np.vectorize(a)
	temp = b(inputs)
	if ret:
		return temp, temp/inputs
	else:
		return b(inputs)

class Layer:
	in_size = 1
	out_size = 1
	weight = np.random.rand(in_size, out_size)
	bias = np.random.rand(out_size)
	activation_f = None
	raw = 0 ### z = sigma(w.a + b)
	out = 0 ### final out
	grad = None



	def __init__(self, in_sz, out_sz, activation=None):
		self.in_size = in_sz
		self.out_size = out_sz
		self.weight = np.random.rand(in_sz, out_sz)
		self.bias = np.random.rand(out_sz)
		if activation is not None:
			self.activation_f = activation

	def forward_(self, input):
		if(len(self.bias.shape) == 2):###To handle the variant batch size, since bias term is of shape(batch size x ouput units)
			self.bias = self.bias[0,:]
		out = np.matmul(input, self.weight) + self.bias
		self.raw = out
		if self.activation_f == None:
			self.out = out
			return self.out
		else:
			# print('Hello')
			self.out, self.grad = self.activation_f(out, True)
			return self.out


if __name__== "__main__":

	c = np.random.rand(1000, 2)
	# d = (2*c[:, 0]).reshape(-1, 1)
	d = (np.sum(c, axis=1)).reshape(-1, 1)
	print("Out shape is", d.shape)
	print(d)

	L1 = Layer(2, 4, activation=sigmoid)
	output = L1.forward_(c)
	L2 = Layer(4, 1)
	out = L2.forward_(output)

	m = Model([L1, L2])
	out = m.forward_pass(c)
	print(out.shape)
	for i in range(1000):
		print(i)
		m.backprop_mse(c, d, alpha= 0.3)
	out1 = m.forward_pass(c)
	l = m.mean_squared_loss(out1, d)
	print(l)
