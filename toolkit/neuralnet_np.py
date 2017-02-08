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


class NeuralNetLearnerNP(SupervisedLearner):

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

			random.shuffle(data)

			for r in range(rows):

				self.resetNet(data[r][0], data[r][1])
				self.forward()
				self.backward()
			
			# self.print(11 - epochs, data[r][0], data[r][1])

			# TODO : criteria for stopping
			# done = curr_acc - prev_acc <= 0.001

			epochs -= 1
			# stopwatch()


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

		self.nets, self.values, self.targets, self.biases, self.errors, self.weights, self.w_delta = [], [], [], [], [], [], []

		for layer in range(total_layers):

			self.nets.append(np.empty((num_neurons[layer])))
			self.values.append(np.empty((num_neurons[layer])))
			self.targets.append(np.empty((num_neurons[layer])))
			self.biases.append(np.ones((num_neurons[layer])))
			self.errors.append(np.empty((num_neurons[layer])))

			if layer < last_layer:
				web = np.random.normal(0.0, 0.3, (num_neurons[layer],num_neurons[layer+1]))
				self.weights.append(np.around(web,3))
				self.w_delta.append(np.zeros((num_neurons[layer],num_neurons[layer+1])))




	def resetNet(self, pattern, target):
		"""
		Initializes values of input neurons
		Initializes target values for output neurons
		TODO : make sure previous pattern's values are properly overwritten (not used)
		"""
		self.values[0] = pattern
		
		for n in range(num_neurons[-1]):		# last layer
			self.targets[-1][n] = 1 if n == target else 0



	def forward(self):

		for curr_layer in range(1, total_layers):
			prev_layer = web = curr_layer - 1

			for n in range(num_neurons[curr_layer]):
				self.nets[curr_layer][n] = np.sum(self.values[prev_layer] * self.weights[web][:,n]) + self.biases[curr_layer][n]

			self.values[curr_layer] = 1 / (1 + np.exp(self.nets[curr_layer]))


	def backward(self):

		global c

		# BACKPROPOGATE ERROR

		for curr_layer in range(last_layer, -1, -1):

			prev_layer = curr_layer + 1
			web = curr_layer

			if curr_layer == last_layer:
				self.errors[curr_layer] = (self.targets[curr_layer] - self.values[curr_layer]) * self.values[curr_layer] * (1 - self.values[curr_layer])

			else:
				for n in range(num_neurons[curr_layer]):
					self.errors[curr_layer][n] = np.sum(self.errors[prev_layer] * self.weights[web][n,:]) * self.values[curr_layer][n] * (1 - self.values[curr_layer][n])

			self.biases[curr_layer] = self.biases[curr_layer] * c * self.errors[curr_layer]				

		# UPDATE W_DELTA AND WEIGHTS

		for web in range(hidden_layers, -1, -1):
			for r in range(num_neurons[web]):
				for c in range(num_neurons[web+1]):
					self.w_delta[web][r][c] = c * self.values[web][r] * self.errors[web+1][c]
			self.weights[web] += self.w_delta[web]



	# def printNeurons(self):
	# 	for layer in range(total_layers):
	# 		for n in self.neurons[layer]:
	# 			print(n.values())
	# 		print("")



	# def print(self, epochNum, pattern, labels):

	# 	print(str(epochNum), pattern, labels)
	# 	for layer in self.neurons:
	# 		for n in layer:
	# 			print(n.value)



	def predict(self, features, labels):
		
		# del labels[:]
		# labels += self.labels

		self.resetNet(features, labels)
		self.forward()

		# print(np.argmax(self.values[last_layer]))


		# TODO : what do I return?
		# return [0]
		return [np.argmax(self.values[last_layer])]