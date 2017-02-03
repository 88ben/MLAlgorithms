from __future__ import (absolute_import, division, print_function, unicode_literals)

from .supervised_learner import SupervisedLearner
from .matrix import Matrix
import numpy as np

nodes_per_layer = np.array([-1, 4, -1])
hidden_layers = len(nodes_per_layer) - 2
total_layers = hidden_layers + 2
last_layer = hidden_layers + 1


class NeuralNetLearner(SupervisedLearner):

	def __init__(self):
		pass

	def train(self, features, labels):

		global nodes_per_layer, hidden_layers, total_layers, last_layer

		nodes_per_layer[[0,-1]] = features.cols, labels.value_count(0)
		mean, stdev = 0.0, 0.3
		# webs = [np.random.normal(mean, stdev, (, )) for layer in range(total_layers)]

 
	def predict(self, features, labels):
		
		# del labels[:]
		# labels += self.labels

		return [0]

	class Node(object):

		def __init__(self, layer, number):
			self.layer = layer
			self.number = number

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