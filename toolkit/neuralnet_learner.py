from __future__ import (absolute_import, division, print_function, unicode_literals)

from .supervised_learner import SupervisedLearner
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

		self.init_webs(features, labels)
		print(self.webs)
 
	def predict(self, features, labels):
		
		# del labels[:]
		# labels += self.labels

		return [0]

	def init_webs(self, features, labels):
		global num_nodes
		# TODO : select best features...
		num_nodes[0], num_nodes[-1] = features.cols, labels.value_count(0)
		web_sizes = [(num_nodes[layer],num_nodes[layer+1]) for layer in range(last_layer)]
		self.webs = [np.random.normal(0.0, 0.1, web) for web in web_sizes]


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