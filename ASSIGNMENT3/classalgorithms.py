from __future__ import division  # floating point division
import numpy as np
import utilities as utils
import math

class Classifier:
	"""
	Generic classifier interface; returns random classification
	Assumes y in {0,1}, rather than {-1, 1}
	"""

	def __init__( self, parameters={} ):
		""" Params can contain any useful parameters for the algorithm """
		self.params = {}

	def reset(self, parameters):
		""" Reset learner """
		self.resetparams(parameters)

	def resetparams(self, parameters):
		""" Can pass parameters to reset with new parameters """
		try:
			utils.update_dictionary_items(self.params,parameters)
		except AttributeError:
			# Variable self.params does not exist, so not updated
			# Create an empty set of params for future reference
			self.params = {}

	def getparams(self):
		return self.params

	def learn(self, Xtrain, ytrain):
		""" Learns using the traindata """

	def predict(self, Xtest):
		probs = np.random.rand(Xtest.shape[0])
		ytest = utils.threshold_probs(probs)
		return ytest

class LinearRegressionClass(Classifier):
	"""
	Linear Regression with ridge regularization
	Simply solves (X.T X/t + lambda eye)^{-1} X.T y/t
	"""
	def __init__( self, parameters={} ):
		self.params = {'regwgt': 0.01}
		self.reset(parameters)

	def reset(self, parameters):
		self.resetparams(parameters)
		self.weights = None

	def learn(self, Xtrain, ytrain):
		""" Learns using the traindata """
		# Ensure ytrain is {-1,1}
		yt = np.copy(ytrain)
		yt[yt == 0] = -1

		# Dividing by numsamples before adding ridge regularization
		# for additional stability; this also makes the
		# regularization parameter not dependent on numsamples
		# if want regularization disappear with more samples, must pass
		# such a regularization parameter lambda/t
		numsamples = Xtrain.shape[0]
		self.weights = np.dot(np.dot(np.linalg.pinv(np.add(np.dot(Xtrain.T,Xtrain)/numsamples,self.params['regwgt']*np.identity(Xtrain.shape[1]))), Xtrain.T),yt)/numsamples

	def predict(self, Xtest):
		ytest = np.dot(Xtest, self.weights)
		ytest[ytest > 0] = 1
		ytest[ytest < 0] = 0
		return ytest

class NaiveBayes(Classifier):
	""" Gaussian naive Bayes;  """

	def __init__(self, parameters={}):
		""" Params can contain any useful parameters for the algorithm """
		# Assumes that a bias unit has been added to feature vector as the last feature
		# If usecolumnones is False, it should ignore this last feature
		self.params = {'usecolumnones': True}
		self.reset(parameters)

	def reset(self, parameters):
		self.resetparams(parameters)
		self.means = []
		self.stds = []
		self.numfeatures = 0
		self.numclasses = 0

	def learn(self, Xtrain, ytrain):
		"""
		In the first code block, you should set self.numclasses and
		self.numfeatures correctly based on the inputs and the given parameters
		(use the column of ones or not).

		In the second code block, you should compute the parameters for each
		feature. In this case, they're mean and std for Gaussian distribution.
		"""

		self.numclasses = 2 #we checked by np.amax and min and we found out it has only 2 classes
				
		if self.params == "usecolumnoneones":
			self.numfeatures = Xtrain.shape[1]
		else:
			self.numfeatures = Xtrain.shape[1] - 1 #excluding columns of 1	
			noOneXtrain = Xtrain[ : , : -1]
			noOneYtrain = ytrain[ : -1]
				
		origin_shape = (self.numclasses, self.numfeatures)
		self.means = np.zeros(origin_shape)
		self.stds = np.zeros(origin_shape)

		
		if self.params == "usecolumnoneones":
			self.means[1] = Xtrain[ytrain == 1].mean(axis = 0)
			self.stds[1] = Xtrain[ytrain == 1].std(axis = 0)
			self.means[0] = Xtrain[ytrain == 0].mean(axis = 0)
			self.stds[0] = Xtrain[ytrain == 0].std(axis = 0)
		
		else:
			self.means[1] = noOneXtrain[ytrain == 1].mean(axis = 0)
			self.stds[1] = noOneXtrain[ytrain == 1].std(axis = 0)
			self.means[0] = noOneXtrain[ytrain == 0].mean(axis = 0)
			self.stds[0] = noOneXtrain[ytrain == 0].std(axis = 0)
						
		uniqueVals, counter = np.unique(ytrain, return_counts = True)
		class0prob = counter[uniqueVals == 0] / ytrain.shape[0]
		class1prob = counter[uniqueVals == 1] / ytrain.shape[0]
		self.probability = np.array([class0prob, class1prob])
		
		assert self.means.shape == origin_shape
		assert self.stds.shape == origin_shape

	def predict(self, Xtest):
		"""
		Use the parameters computed in self.learn to give predictions on new
		observations.
		"""
		ytest = np.zeros(Xtest.shape[0], dtype=int)
		for i in range(Xtest.shape[0]):
			dataProbability = 1
			for j in range(self.numfeatures):
				dataProbability *= self.gaussianDistribution(Xtest[i][j], self.means[0][j], self.stds[0][j])
			class0prob = self.probability[0] * dataProbability

			dataProbability = 1
			for j in range(self.numfeatures):
				dataProbability *= self.gaussianDistribution(Xtest[i][j], self.means[1][j], self.stds[1][j])
			class1prob = self.probability[1] * dataProbability
		
			if class0prob > class1prob:
				ytest[i] = 0
			else:
				ytest[i] = 1
			
		assert len(ytest) == Xtest.shape[0]
		return ytest
		
	def gaussianDistribution(self, data, mean, std):
		GD = 1 / math.sqrt(2 * math.pi * (std ** 2) ) * math.exp( - ((data - mean) **2 / (2 * (std ** 2) )))
		return GD


class LogitReg(Classifier):

	def __init__(self, parameters={}):
		# Default: no regularization
		self.params = {'regwgt': 0.0, 'regularizer': 'None'}
		self.reset(parameters)

	def reset(self, parameters):
		self.resetparams(parameters)
		self.weights = None
		if self.params['regularizer'] is 'l2':
			self.regularizer = (utils.l2, utils.dl2)
		else:
			self.regularizer = (lambda w: 0, lambda w: np.zeros(w.shape,))

	def logit_cost(self, theta, X, y):
		"""
		Compute cost for logistic regression using theta as the parameters.
		"""

		cost = 0.0
		yHat = utils.sigmoid(np.dot(X, theta)) #we have only twwo classes
		
		llSum = 0
		for i in range(X.shape[0]):
			llSum += y[i] * np.log(yHat[i]) + (1-y[i])*np.log(1-yHat[i])
			
		cost = llSum / X.shape[0] #normalizing by number of features
		
		if (self.params['regwgt'] == 'l2'):
			cost += self.params['regwgt'] * utils.l2(theta)
		
		return cost

	def logit_cost_grad(self, theta, X, y):
		"""
		Compute gradients of the cost with respect to theta.
		"""

		grad = np.zeros(len(theta))

		llSum = 0
		yHat = utils.sigmoid(np.array([np.dot(X, theta)]))
		for j in range(yHat.shape[0]):
			for i in range(theta.shape[0]):
				sum = (yHat[j] - y) * X[i]
				grad[i] = sum
		if (self.params['regwgt'] == 'l2'):
			grad = grad + self.params['regwgt'] * utils.dl2(theta)

		return grad

	def learn(self, Xtrain, ytrain):
		"""
		Learn the weights using the training data
		"""

		self.weights = np.zeros(Xtrain.shape[1],)
		
		#based on notes, using SGD here:
		
		numsamples = Xtrain.shape[0]
		dim = Xtrain.shape[1]
		#copying the same code of ass2 with some changes
		self.weights = np.random.rand(dim) #giving random wieghts at the very first place
		
		epochs = 500
		
		for i in range(epochs):
			shuffling = np.arange(numsamples)
			np.random.shuffle(shuffling)
			stepSize = 0.01 / (i + 1) #decreasing stepsize
			for t in shuffling:
				grad = self.logit_cost_grad(self.weights, Xtrain[t], ytrain[t]) #using BGD function
				self.weights = self.weights - stepSize * grad

	def predict(self, Xtest):
		"""
		Use the parameters computed in self.learn to give predictions on new
		observations.
		"""
		ytest = np.zeros(Xtest.shape[0], dtype=int)
		
		probability = utils.sigmoid(np.dot(self.weights, Xtest.transpose()))
		classifier = np.ones(len(probability),)
		classifier[probability < 0.5] = 0
		ytest = classifier
		

		assert len(ytest) == Xtest.shape[0]
		return ytest

'''
class NeuralNet(Classifier):
""" Implement a neural network with a single hidden layer. Cross entropy is
used as the cost function.

Parameters:
nh -- number of hidden units
transfer -- transfer function, in this case, sigmoid
stepsize -- stepsize for gradient descent
epochs -- learning epochs

Note:
1) feedforword will be useful! Make sure it can run properly.
2) Implement the back-propagation algorithm with one layer in ``backprop`` without
any other technique or trick or regularization. However, you can implement
whatever you want outside ``backprob``.
3) Set the best params you find as the default params. The performance with
the default params will affect the points you get.
"""
def __init__(self, parameters={}):
self.params = {'nh': 16,
'transfer': 'sigmoid',
'stepsize': 0.01,
'epochs': 10}
self.reset(parameters)

def reset(self, parameters):
self.resetparams(parameters)
if self.params['transfer'] is 'sigmoid':
self.transfer = utils.sigmoid
self.dtransfer = utils.dsigmoid
else:
# For now, only allowing sigmoid transfer
raise Exception('NeuralNet -> can only handle sigmoid transfer, must set option transfer to string sigmoid')
self.w_input = None
self.w_output = None

def feedforward(self, inputs):
"""
Returns the output of the current neural network for the given input
"""
# hidden activations
a_hidden = self.transfer(np.dot(self.w_input, inputs))

# output activations
a_output = self.transfer(np.dot(self.w_output, a_hidden))

return (a_hidden, a_output)

def backprop(self, x, y):
"""
Return a tuple ``(nabla_input, nabla_output)`` representing the gradients
for the cost function with respect to self.w_input and self.w_output.
"""

### YOUR CODE HERE

### END YOUR CODE

assert nabla_input.shape == self.w_input.shape
assert nabla_output.shape == self.w_output.shape
return (nabla_input, nabla_output)

# TODO: implement learn and predict functions

class KernelLogitReg(LogitReg):
""" Implement kernel logistic regression.

This class should be quite similar to class LogitReg except one more parameter
'kernel'. You should use this parameter to decide which kernel to use (None,
linear or hamming).

Note:
1) Please use 'linear' and 'hamming' as the input of the paramteter
'kernel'. For example, you can create a logistic regression classifier with
linear kerenl with "KernelLogitReg({'kernel': 'linear'})".
2) Please don't introduce any randomness when computing the kernel representation.
"""
def __init__(self, parameters={}):
# Default: no regularization
self.params = {'regwgt': 0.0, 'regularizer': 'None', 'kernel': 'None'}
self.reset(parameters)

def learn(self, Xtrain, ytrain):
"""
Learn the weights using the training data.

Ktrain the is the kernel representation of the Xtrain.
"""
Ktrain = None

### YOUR CODE HERE

### END YOUR CODE

self.weights = np.zeros(Ktrain.shape[1],)

### YOUR CODE HERE

### END YOUR CODE

self.transformed = Ktrain # Don't delete this line. It's for evaluation.

# TODO: implement necessary functions


# ======================================================================

def test_lr():
print("Basic test for logistic regression...")
clf = LogitReg()
theta = np.array([0.])
X = np.array([[1.]])
y = np.array([0])

try:
cost = clf.logit_cost(theta, X, y)
except:
raise AssertionError("Incorrect input format for logit_cost!")
assert isinstance(cost, float), "logit_cost should return a float!"

try:
grad = clf.logit_cost_grad(theta, X, y)
except:
raise AssertionError("Incorrect input format for logit_cost_grad!")
assert isinstance(grad, np.ndarray), "logit_cost_grad should return a numpy array!"

print("Test passed!")
print("-" * 50)

def test_nn():
print("Basic test for neural network...")
clf = NeuralNet()
X = np.array([[1., 2.], [2., 1.]])
y = np.array([0, 1])
clf.learn(X, y)

assert isinstance(clf.w_input, np.ndarray), "w_input should be a numpy array!"
assert isinstance(clf.w_output, np.ndarray), "w_output should be a numpy array!"

try:
res = clf.feedforward(X[0, :])
except:
raise AssertionError("feedforward doesn't work!")

try:
res = clf.backprop(X[0, :], y[0])
except:
raise AssertionError("backprob doesn't work!")

print("Test passed!")
print("-" * 50)
'''
def main():
	test_lr()
	test_nn()

	if __name__ == "__main__":
		main()
