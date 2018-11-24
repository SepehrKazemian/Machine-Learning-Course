from __future__ import division  # floating point division
import numpy as np
import utilities as utils
import script_classify as scs
import math
import time

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
				
		if self.params["usecolumnones"]:
			self.numfeatures = Xtrain.shape[1]
		else:
			self.numfeatures = Xtrain.shape[1] - 1 #excluding columns of 1	
			noOneXtrain = Xtrain[ : , : -1]
			noOneYtrain = ytrain[ : -1]
				
		origin_shape = (self.numclasses, self.numfeatures)
		self.means = np.zeros(origin_shape)
		self.stds = np.zeros(origin_shape)

		
		if self.params["usecolumnones"]:
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
		if std != 0:
			GD = 1 / math.sqrt(2 * math.pi * (std ** 2) ) * math.exp( - ((data - mean) **2 / (2 * (std ** 2) )))
			return GD
		if std == 0:
			return 1


class LogitReg(Classifier):

	def __init__(self, parameters={}):
		# Default: no regularization
		self.params = {'stepsize': 0.01, 'epochs': 100, 'regwgt': 0.0, 'regularizer': 'None'}
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
		yHat = utils.sigmoid(np.dot(X.transpose(), theta)) #we have only twwo classes
		
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
		stepSize = self.params['stepsize']
		
		for i in range(self.params['epochs']):
			shuffling = np.arange(numsamples)
			np.random.shuffle(shuffling)
#			stepSize = stepSize / (i + 1) #decreasing stepsize
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
	
#******** using http://www.wildml.com/2015/09/implementing-a-neural-network-from-scratch/ *****#
	def __init__(self, parameters={}):
		self.params = {'nh': 16,
		'transfer': 'sigmoid',
		'stepsize': 0.05,
		'epochs': 10,
		'hiddenLayers' : 1}
		self.reset(parameters)

	def reset(self, parameters):
		self.resetparams(parameters)
		if self.params['transfer'] is 'sigmoid':
#			print("aaa")
			self.transfer = utils.sigmoid
			self.dtransfer = utils.dsigmoid
		else:
			# For now, only allowing sigmoid transfer
			raise Exception('NeuralNet -> can only handle sigmoid transfer, must set option transfer to string sigmoid')
		self.w_input = None
		self.w2 = None
		self.w_output = None

	def feedforward(self, inputs, numberOfHiddenLayer):
		"""
		Returns the output of the current neural network for the given input
		"""
		
		
		# hidden activations
		a1 = self.transfer(np.dot(self.w_input, inputs))

		# output activations

		a2 = self.transfer(np.dot(self.w2, a1))
		
		if numberOfHiddenLayer == 2:

			a3 = self.transfer(np.dot(self.w_output, a2))
			return(a1, a2, a3)
		
		elif numberOfHiddenLayer == 1:
			return (a1, a2)
	

	def backprop(self, x, y):
		"""
		Return a tuple ``(nabla_input, nabla_output)`` representing the gradients
		for the cost function with respect to self.w_input and self.w_output.
		"""
		if self.params['hiddenLayers'] == 2:
			numberOfHiddenLayer = 2
			x = x.reshape((x.shape[0], 1)) #unless we will get numpy.float64 error
			(a1, a2, output) = self.feedforward(x, numberOfHiddenLayer)
#			print(x.shape)
#			print(a1.shape)
#			print(a2.shape)
#			print(output.shape)

			
			bp = output - y
#			print(a2.shape)
			nabla_output = (np.dot(bp, a2.transpose()))/x.shape[0] #normalizing
			
			bp1 = np.multiply(np.dot(self.w_output.transpose(), bp), self.dtransfer(np.dot(self.w2, a1)))
			nabla1 = (np.dot(bp1, a1.transpose()))/x.shape[0] #normalizing

			bp2 = np.multiply(np.dot(self.w2.transpose(), bp1), self.dtransfer(np.dot(self.w_input, x)))			
			nabla_input = (np.dot(bp2, x.transpose()))/x.shape[0] #normalizing

			
			
			assert nabla_input.shape == self.w_input.shape
			assert nabla1.shape == self.w2.shape
			assert nabla_output.shape == self.w_output.shape
			return (nabla_input, nabla1, nabla_output)
			
		elif self.params['hiddenLayers'] == 1:
			numberOfHiddenLayer = 1
			
			x = x.reshape((x.shape[0], 1)) #unless we will get numpy.float64 error
			(layer1, output) = self.feedforward(x, numberOfHiddenLayer)
			bp1 = output - y
			nabla_output = (np.dot(bp1, layer1.transpose()))/x.shape[0] #normalizing
			bp2 = (np.dot(self.w2.transpose(), bp1) * self.dtransfer(np.dot(self.w_input, x)))
			
			nabla_input = (np.multiply(bp2, x.transpose()))/x.shape[0] #normalizing

			assert nabla_input.shape == self.w_input.shape
			assert nabla_output.shape == self.w2.shape
			return (nabla_input, nabla_output)


	def learn(self, Xtrain, Ytrain):
		dim = Xtrain.shape[1]
#		print(dim)
		if self.params['hiddenLayers'] == 1:
			self.w_input = np.random.rand(self.params['nh'], dim)
			self.w2 = np.random.rand(1, self.params['nh'])
		elif self.params['hiddenLayers'] == 2:
			self.w_input = np.random.rand(self.params['nh'], dim)
			self.w2 = np.random.rand(self.params['nh'], self.params['nh'])
			self.w_output = np.random.rand(1, self.params['nh'])
		numsamples = Xtrain.shape[0]
#		print(numsamples)
		
		if self.params['hiddenLayers'] == 1:
			for i in range(self.params['epochs']):
				shuffling = np.arange(numsamples)
				np.random.shuffle(shuffling)
	#			self.params['stepsize'] = self.params['stepsize'] / (i)
	#			I wanted to decrease the stepsize, but it increased the error
				counterr = 0
				for miniBatches in shuffling:
					nabla_input, nabla_output = self.backprop(Xtrain[miniBatches], Ytrain[miniBatches])
					self.w_input = np.subtract(self.w_input, self.params['stepsize'] * nabla_input)
					self.w2 = np.subtract(self.w2, self.params['stepsize'] * nabla_output)
					test1 = self.transfer(np.dot(self.w_input, Xtrain[miniBatches]))
					test2 = self.transfer(np.dot(self.w2, test1))
					err1 = (math.fabs(Ytrain[miniBatches]) > 0.5)
					err2 = (math.fabs(test2) > 0.5)
					if (err1 == err2):
						counterr += 1
					
#				elif self.params['hiddenLayers'] == 2:
#					nabla_input, nabla1, nabla_output = self.backprop(Xtrain[miniBatches], Ytrain[miniBatches])
#					self.w_input = np.subtract(self.w_input, self.params['stepsize'] * nabla_input)
#					self.w2 = np.subtract(self.w2, self.params['stepsize'] * nabla1)
#					self.w_output = np.subtract(self.w_output, self.params['stepsize'] * nabla_output)
#					test1 = self.transfer(np.dot(self.w_input, Xtrain[miniBatches]))
#					test2 = self.transfer(np.dot(self.w2, test1))
#					test3 = self.transfer(np.dot(self.w_output, test2))
#					err1 = (math.fabs(Ytrain[miniBatches]) > 0.5)
#					err2 = (math.fabs(test3) > 0.5)
#					if (err1 == err2):
#						counterr += 1
#						print(counterr / Ytrain.shape[0])

		e = 1e-3
		p = 0.9
		if self.params['hiddenLayers'] == 2:
			for i in range(self.params['epochs']):
				eta = 0.001
				v_input = np.zeros((self.params['nh'], dim))
				v1 = np.zeros((self.params['nh'], self.params['nh']))
				v_output = np.zeros((1, self.params['nh']))
				shuffling = np.arange(numsamples)
				np.random.shuffle(shuffling)
				for j in shuffling:
					nabla_input, nabla1, nabla_output = self.backprop(Xtrain[j], Ytrain[j])
					v_input = p * v_input + (1 - p) * (nabla_input**2)
					v1 = p * v1 + (1 - p) * (nabla1**2)
					v_output = p * v_output + (1 - p) * (nabla_output**2)
					self.w_input = self.w_input - (self.params['stepsize'] / np.sqrt(v_input+ e) * nabla_input )
					self.w2 = self.w2 - (self.params['stepsize'] / np.sqrt(v1+ e) * nabla1 )
					self.w_output = self.w_output - (self.params['stepsize'] / np.sqrt(v_output+ e) * nabla_output )
				
				
	def predict(self, Xtest):
		ytest = np.zeros(Xtest.shape[0], dtype=int)
		
#		print(self.feedforward(Xtest[i, :], self.params['hiddenLayers']))
#		print(self.feedforward(Xtest[0, :], self.params['hiddenLayers']))
		ytest = [(self.feedforward(Xtest[i, :], self.params['hiddenLayers'])[-1] > 0.5) for i in range(Xtest.shape[0])]
		
#		print(ytest)
		
		assert len(ytest) == Xtest.shape[0]
		return ytest
				
				


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
		objLR = LogitReg()
		"""
		Learn the weights using the training data.

		Ktrain the is the kernel representation of the Xtrain.
		"""
		Ktrain = None

		### YOUR CODE HERE
		self.kNum = 50
		self.kernel = Xtrain[ : self.kNum]
		
		Ktrain = np.zeros((Xtrain.shape[0], self.kNum))
		
		if self.params['kernel'] == 'linear':
			Ktrain = np.dot(Xtrain, self.kernel.T)
		elif self.params['kernel'] == 'hamming':
		#	print("ccc")
			Ktrain = self.hammingDis(Xtrain)
						
						
		

		### END YOUR CODE

		counter = 0
		trueVar = 0
		self.weights = np.zeros(Ktrain.shape[1],)

		### YOUR CODE HERE
		#using SGD --> copying my SGD algorithm and modifying it
		numsamples = Xtrain.shape[0]
		self.weights = np.random.rand(self.kNum)
		if self.params['kernel'] == 'hamming':
			stepSize = 0.001
		else:
			stepSize = 0.01
		for i in range(700):
			shuffling = np.arange(numsamples)
			np.random.shuffle(shuffling)
			if (self.params['kernel'] == 'hamming'):
				stepSize /= (i+1)
			
			for miniBatches in shuffling:
				g = objLR.logit_cost_grad(self.weights, Ktrain[miniBatches], ytrain[miniBatches])
				self.weights = self.weights - stepSize * g
			
				counter += 1
				err1 = np.dot(self.weights, Ktrain[miniBatches].T) > 0.5
				err2 = ytrain[miniBatches] > 0.5
				if err1 == err2:
					trueVar += 1

			print(trueVar / counter)
				
		### END YOUR CODE

		self.transformed = Ktrain # Don't delete this line. It's for evaluation.

		# TODO: implement necessary functions
	def predict(self,Xtest):
	
		if self.params['kernel'] == 'linear':
			kernelTest = np.dot(Xtest, self.kernel.T) 
		elif self.params['kernel'] == 'hamming':
		#	print("bbb")
			kernelTest = self.hammingDis(Xtest)
			
		testYnoSig = np.dot(self.weights, kernelTest.transpose())
		testYprobability = utils.sigmoid(testYnoSig)
		
		ytest = utils.threshold_probs(testYprobability)
		
		assert len(ytest) == Xtest.shape[0]
		return ytest

	def hammingDis(self, Xtrain):
		Ktrain = None
		Ktrain = np.zeros((Xtrain.shape[0], self.kNum))
	#	print(Xtrain.shape)

		for i in range(Xtrain.shape[0]):
			for k in range(self.kNum):
				hamDis = 0
				for j in range(len(Xtrain[i])):
					if Xtrain[i][j] == Xtrain[k][j]:
						Ktrain[i, k] += 1	
						
	#	print(Ktrain)
		return Ktrain
		
		
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

def main():
	test_lr()
	test_nn()

	if __name__ == "__main__":
		main()
