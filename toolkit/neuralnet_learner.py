from __future__ import (absolute_import, division, print_function, unicode_literals)

from .supervised_learner import SupervisedLearner
from .matrix import Matrix
import numpy

hidden_layers = 1
nodes_per_layer = [-1, 4, -1]
total_layers = hidden_layers + 2
last_layer = hidden_layers + 1


class NeuralNetLearner(SupervisedLearner):
	"""
	For nominal labels, this model simply returns the majority class. For
	continuous labels, it returns the mean value.
	If the learning model you're using doesn't do as well as this one,
	it's time to find a new learning model.
	"""

	labels = []

	def __init__(self):
		pass

	def train(self, features, labels):
		
		# create webs

		# webs = [np.random.normal(0.0, 0.3, (web, )) for web in range(total_layers-1)]
		
 
	def predict(self, features, labels):
		
		# del labels[:]
		# labels += self.labels

		pass

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