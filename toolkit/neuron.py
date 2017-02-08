import numpy as np

class Neuron(object):

	def __init__(self, layer, index, net=None, value=None, target=None, bias=1.0, error=None):
		self.layer = layer
		self.index = index
		self.net = net
		self.value = value
		self.target = target
		self.bias = bias
		self.error = error


	def values(self):
		return self.layer, self.index, self.value, self.target, self.bias, self.error



	def updateBias(self, c):

		self.bias += self.bias * c * self.error



	def calcError(self, inherited_error):
		"""
		This is the error propogated back to fine-tune the weights
		"""
		if inherited_error == None:		# output neurons
			self.error = (self.target - self.value) * self.activate_der()
		else:
			self.error = inherited_error * self.activate_der()



	def activate_der(self):
		"""
		The derivitive of the activation function is used in calculating error
		"""
		return self.value * (1 - self.value)



	def activate(self, net):
		"""
		Input neurons are NOT activated (since no net value enters them)
		Hidden and Output neurons ARE activated from previous layer's net
		"""
		if self.layer != 0:
			self.net = net
			self.value = 1 / (1 + np.exp(net))
		# else:
			# should I throw an error?