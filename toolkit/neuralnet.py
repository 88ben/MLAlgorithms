from __future__ import absolute_import, division, print_function, unicode_literals
from .supervised import SupervisedLearner
from .neuron import Neuron
from .matrix import Matrix
from .my import stopwatch
import numpy as np
import random


c = 0.1
num_neurons = [-1, 4, -1]
hidden_layers = len(num_neurons) - 2
total_layers = hidden_layers + 2
last_layer = hidden_layers + 1


class NeuralNetLearner(SupervisedLearner):

	def __init__(self):
		pass

	def train(self, features, labels):

		# TODO : how do I handle the bias?

		# done = False
		epochs = 10
		rows = features.rows
		inputs = features.cols

		self.createNet(features, labels)

		data = list(zip(features.data, labels.data))

		while epochs:

			# features.shuffle(labels)
			# patterns = np.copy(features.data)
			random.shuffle(data)
			correct = 0

			for r in range(rows):

				# self.resetNet(patterns[r], labels.row(r))
				self.resetNet(data[r][0], data[r][1])
				self.forward()
				self.backward()
			# self.print(11 - epochs, data[r][0], data[r][1])

			# TODO : criteria for stopping
			# done = curr_acc - prev_acc <= 0.001

			epochs -= 1


	def createNet(self, features, labels):
		"""
		Sets number of neurons per layer (except preset hidden layers)
		Randomly sets weights (gaussian, mean=0, std=0.3)
		Sets change in weights to 0
		Creates neurons for all layers
		"""

		# TODO : select only good features
		num_neurons[0] = features.cols
		num_neurons[-1] = labels.value_count(0)

		# NODES

		self.neurons = []
		for layer in range(total_layers):
			self.neurons.append(np.array([Neuron(layer,n) for n in range(num_neurons[layer])]))

		# WEIGHTS

		self.weights = []
		for layer in range(last_layer):
			web = np.random.normal(0.0, 0.3, (num_neurons[layer],num_neurons[layer+1]))
			self.weights.append(np.around(web,3))
			# self.w_delta.append(np.array([[0]*num_neurons[layer+1]]*num_neurons[layer]))



	def resetNet(self, pattern, target):
		"""
		Initializes values of input neurons
		Initializes target values for output neurons
		TODO : make sure previous pattern's values are properly overwritten (not used)
		"""

		for n in range(len(self.neurons[0])):		# first layer
			self.neurons[0][n].value = pattern[n]
		
		for n in range(len(self.neurons[-1])):		# last layer
			self.neurons[-1][n].target = 1 if n == target else 0



	def forward(self):

		for curr_layer in range(1, total_layers):

			prev_layer = web = curr_layer - 1
			values = self.getLayerValues(prev_layer)

			for n in range(num_neurons[curr_layer]):

				weights = self.weights[web][:,n]
				net = np.sum(values * weights) + self.neurons[curr_layer][n].bias
				self.neurons[curr_layer][n].activate(net)



	def backward(self):

		global c

		# BACKPROPOGATE ERROR

		for curr_layer in range(last_layer, -1, -1):

			prev_layer = curr_layer + 1
			web = curr_layer
			errors = self.getLayerErrors(prev_layer)

			for n in range(num_neurons[curr_layer]):

				if prev_layer == total_layers:	# output layer
					inherited_error = None
				else:							# all other layers
					weights = self.weights[web][n,:]
					inherited_error = np.sum(errors * weights)
				
				self.neurons[curr_layer][n].calcError(inherited_error)
				self.neurons[curr_layer][n].updateBias(c)

		# UPDATE W_DELTA AND WEIGHTS

		for web in range(hidden_layers, -1, -1):
			for r in range(num_neurons[web]):
				for c in range(num_neurons[web+1]):
					# self.w_delta[web][r][c] = c * self.neurons[web][r].value * self.neurons[web+1][c].error
					# self.weights[web][r][c] += self.w_delta[web][r][c]
					self.weights[web][r][c] += c * self.neurons[web][r].value * self.neurons[web+1][c].error



	def getLayerValues(self, layer):
		"""
		To calculate net value for neuron, you need ALL values of previous layer
		"""
		return np.array([n.value for n in self.neurons[layer]])



	def getLayerErrors(self, layer):
		if layer > last_layer:
			return None
		else:
			return np.array([n.error for n in self.neurons[layer]])



	# def printNeurons(self):
	# 	for layer in range(total_layers):
	# 		for n in self.neurons[layer]:
	# 			print(n.values())
	# 		print("")



	def print(self, epochNum, pattern, labels):

		print(str(epochNum), pattern, labels)
		# for layer in self.neurons:
		# 	print([n.value for n in layer])
		for web in self.weights:
			print(web)



	def predict(self, features, labels):
		
		# del labels[:]
		# labels += self.labels

		self.resetNet(features, labels)
		self.forward()

		# return [0]
		return [np.argmax(self.getLayerValues(last_layer))]