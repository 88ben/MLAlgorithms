from __future__ import (absolute_import, division, print_function, unicode_literals)

from .supervised import SupervisedLearner
from .matrix import Matrix
import numpy as np


class PerceptronLearner(SupervisedLearner):

	def __init__(self):
		pass

	def train(self, features, labels):

		c = 0.1;
		curr_acc = 0.0
		prev_acc = -1.0

		epochs = 0
		done = False
		not_better = 0
		inputs = features.cols + 1 # includes bias
		rows = features.rows

		weights = np.zeros(inputs,dtype=float)
		w_delta = np.zeros(inputs,dtype=float)

		while not done:

			features.shuffle(labels)
			correct = 0

			patterns = np.concatenate((np.copy(features.data),np.ones((rows,1))),axis=1)

			for r in range(rows):

				net = np.sum(np.multiply(patterns[r],weights))
				target = np.array(labels.row(r))
				output = 1 if net > 0 else 0
				w_delta = c * (target - output) * patterns[r]
				weights += w_delta

				if target == output: correct += 1

			curr_acc = correct / rows
			done = curr_acc - prev_acc <= 0.001
			prev_acc = curr_acc
			epochs += 1

		self.weights = weights


	def predict(self, features, labels):
		"""
		:type features: [float]
		:type labels: [float]
		"""
		# I changed supervised.py to receive a return value as the prediction but only in the else statement
		# probably need to remember that
		net = np.sum(np.array(features) * self.weights[:-1])
		net += self.weights[-1:]
		labels = [0]
		labels[0] = 1 if net > 0 else 0
		return labels