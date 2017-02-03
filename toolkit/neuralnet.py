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

		self.init_weights(features, labels)
		self.init_nodes(features, labels)

 
	def predict(self, features, labels):
		
		# del labels[:]
		# labels += self.labels

		return [0]

	def init_weights(self, features, labels):
		# TODO : remove useless features before setting number of nodes in first layer
		self.weights = []
		num_nodes[0] = features.cols
		num_nodes[-1] = labels.value_count(0)

		for layer in range(last_layer):
			self.weights.append(np.random.normal(0.0, 0.3, (num_nodes[layer],num_nodes[layer+1])))

	def init_nodes(self, features, labels):

		self.nodes = []

		for layer in range(total_layers):
			group = []
			for node in range(num_nodes[layer]):
				group.append(Node(layer,node))
			self.nodes.append(group)

		# self.print_nodes()

	def print_nodes(self):
		for layer in range(total_layers):
			for node in self.nodes[layer]:
				print(node.values())
			print("")


class Node(object):

	def __init__(self, layer, number, bias=1):
		self.layer = layer
		self.number = number
		self.bias = bias

	def values(self):
		return self.layer, self.number

	def calc_w_delta(c, o, error):
		return c * o * error

	def calc_error(target, output):
		global last_layer
		if self.layer == last_layer:
			return (target - output) * output * (1 - output)
		else:
			pass
			# summation of all ((child error * weight to that child) * output * (1 - output))

	def calc_output(net):
		return 1 / (1 + np.exp(net))