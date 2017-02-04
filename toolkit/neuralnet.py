from __future__ import (absolute_import, division, print_function, unicode_literals)

from .supervised import SupervisedLearner
from .matrix import Matrix
import numpy as np

num_nodes = [-1, 4, -1]
hidden_layers = len(num_nodes) - 2
total_layers = hidden_layers + 2
last_layer = hidden_layers + 1


class NeuralNetLearner(SupervisedLearner):

	def __init__(self):
		pass

	def train(self, features, labels):

		# TODO : how do I handle the bias?

		c = 0.1
		done = False
		bias = 1.0
		rows = features.rows
		inputs = features.cols	# bias?

		self.create_net(features, labels)

		while not done:

			features.shuffle(labels)
			correct = 0

			patterns = np.copy(features.data)	# bias?

			for r in range(rows):

				self.initialize_IO(patterns[r], labels.row(r))

				for layer in range(total_layers):
					print(layer)

					for n in self.nodes[layer+1]:
						# print(self.weights[layer][:,0])
						# print(np.sum(self.getLayerValues(layer) * self.weights[layer][:,0]))
						self.nodes[layer+1][0].output = np.sum(self.getLayerValues(layer) * self.weights[layer][:,0])
					# print(self.weights[layer])

			# done = curr_acc - prev_acc <= 0.001
			break

		# output = features.get(0,node) if layer == 0 else None

 
	def predict(self, features, labels):
		
		# del labels[:]
		# labels += self.labels

		return [0]

	def initialize_IO(self, pattern, target):

		for n in range(len(self.nodes[0])):		# first layer
			self.nodes[0][n].output = pattern[n]
		
		for n in range(len(self.nodes[-1])):	# last layer
			self.nodes[-1][n].target = 1 if n == target else 0

	def create_net(self, features, labels):

		# TODO : select good features
		num_nodes[0] = features.cols
		num_nodes[-1] = labels.value_count(0)

		# WEIGHTS

		self.weights = []
		for layer in range(last_layer):
			web = np.random.normal(0.0, 0.3, (num_nodes[layer],num_nodes[layer+1]))
			self.weights.append(np.around(web,3))

		# DELTA WEIGHTS

		self.w_delta = []
		for layer in range(last_layer):
			self.w_delta.append(np.array([[0]*num_nodes[layer+1]]*num_nodes[layer]))

		# NODES

		self.nodes = []
		for layer in range(total_layers):
			self.nodes.append(np.array([Node(layer,node) for node in range(num_nodes[layer])]))

		# self.print_nodes()

	def getLayerValues(self, layer):
		return np.array([n.output for n in self.nodes[layer]])

	def print_nodes(self):
		for layer in range(total_layers):
			for node in self.nodes[layer]:
				print(node.values())
			print("")


class Node(object):

	def __init__(self, layer, number, target=None, output=None, bias=1.0):
		self.layer = layer
		self.number = number
		self.target = target
		self.output = output
		self.bias = bias

	def values(self):
		return self.layer, self.number, self.target, self.output, self.bias

	def calc_w_delta(c, output, error):
		return c * output * error

	def calc_error(target, output):
		global last_layer
		if self.layer == last_layer:
			return (target - output) * der_output(target)
		else:
			pass
			# summation of all ((child error * weight to that child) * output * (1 - output))

	def get_output(net):
		if self.layer == 0:
			return self.output
		else:
			return 1 / (1 + np.exp(net))

	def der_output(net):
		return net * (1 - net)