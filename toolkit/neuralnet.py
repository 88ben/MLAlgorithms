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

		c = 0.1

		self.create_net(features, labels)

		# output = features.get(0,node) if layer == 0 else None

 
	def predict(self, features, labels):
		
		# del labels[:]
		# labels += self.labels

		return [0]

	def create_net(self, features, labels):

		# TODO : select good features
		num_nodes[0] = features.cols
		num_nodes[-1] = labels.value_count(0)

		# WEIGHTS

		self.weights = []
		for layer in range(last_layer):
			self.weights.append(np.around(np.random.normal(0.0, 0.3, (num_nodes[layer],num_nodes[layer+1])),3).tolist())

		# CHANGE IN WEIGHT

		self.w_delta = []
		for layer in range(last_layer):
			self.w_delta.append([[0]*num_nodes[layer+1]]*num_nodes[layer])

		# NODES

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

	def __init__(self, layer, number, output=None, bias=1.0):
		self.layer = layer
		self.number = number
		self.output = output
		self.bias = bias

	def values(self):
		return self.layer, self.number, self.output, self.bias

	def calc_w_delta(c, o, error):
		return c * o * error

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