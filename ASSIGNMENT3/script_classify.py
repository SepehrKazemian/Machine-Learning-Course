from __future__ import division  # floating point division
import csv
import random
import math
import numpy as np
from termcolor import colored

import dataloader as dtl
import classalgorithms as algs


def getaccuracy(ytest, predictions):
	correct = 0
	for i in range(len(ytest)):
		if ytest[i] == predictions[i]:
			correct += 1
	return (correct/float(len(ytest))) * 100.0

def geterror(ytest, predictions):
	return (100.0-getaccuracy(ytest, predictions))

## k-fold cross-validation
#*************************************KFOLD CODE IS WRITTEN IN MAIN CLASS*************************



if __name__ == '__main__':
	trainsize = 5000
	testsize = 5000
	numruns = 1
	k_foldStratified = 0
	k_foldClass = ""
	
	k_foldClass = input("do you want to run the code with K-Fold cross validation? Y/N")
	if k_foldClass == "Y":
		k_foldClass = 1
		k_foldStratified = 0
	elif k_foldClass == "N":
		k_foldStratified = input("do you want to run the code with K-Fold stratified cross validation? Y/N")
		if k_foldStratified == "Y":
			k_foldClass = 0
			k_foldStratified = 1
	
	else:
		k_foldStratified = 0
		k_foldClass = 0
	
	

	classalgs = {'Random': algs.Classifier(),
#				'Naive Bayes': algs.NaiveBayes({'usecolumnones': False}),
 # 			   'Naive Bayes Ones': algs.NaiveBayes({'usecolumnones': True}),
  			   'Linear Regression': algs.LinearRegressionClass(),
   			  'Logistic Regression': algs.LogitReg(),
				 'Neural Network': algs.NeuralNet({'hiddenLayers' : 1}),
				  'Neural Network': algs.NeuralNet({'epochs': 100, 'hiddenLayers' : 2}),
				  'Kernel Logistic Regression linear': algs.KernelLogitReg({'kernel': 'linear', 'regularizer': 'l2'}),
				  'Kernel Logistic Regression hamming': algs.KernelLogitReg({'kernel': 'hamming', 'regularizer': 'l2'})

				}


	
	k_foldClass = 0
	k_foldStratified = 1
	
	numalgs = len(classalgs)
	k_fold = 5
	parameters = (
		{'nh': 4, 'epochs': 100, 'stepsize': 0.01, "regularizer" : "l2"},
		{'nh': 8, 'epochs': 500, 'stepsize': 0.01, "regularizer" : "None"},
		{'nh': 4, 'epochs': 500, 'stepsize': 0.04, "regularizer" : "l2"},
		{'nh': 8, 'epochs': 100, 'stepsize': 0.04, "regularizer" : "None"},
		{'nh': 16, 'epochs': 100, 'stepsize': 0.01, "regularizer" : "None"},
		{'nh': 16, 'epochs': 100, 'stepsize': 0.04, "regularizer" : "None"},
		)
	numparams = len(parameters)

	CVError = {}
	errors = {}
	for learnername in classalgs:
		errors[learnername] = np.zeros((numparams,numruns))

	if k_fold != 0 and k_foldClass == 1:
		for learnername in classalgs:
			CVError[learnername] = np.zeros((numparams, numruns))
		


		for r in range(numruns):
			trainset, testset = dtl.load_susy(trainsize,testsize)
			#trainset, testset = dtl.load_susy_complete(trainsize,testsize)
			#trainset, testset = dtl.load_census(trainsize,testsize)

			
			print(('Running on train={0} and test={1} samples for run {2}').format(trainset[0].shape[0], testset[0].shape[0],r))

			for learnername, learner in classalgs.items():

				meanErrParam = []
				nameParam = []
				params = ""
#				print(numparams)
				for p in range(numparams):
					
					params = parameters[p]
					learner.reset(params)
					print ('Running learner = ' + learnername + ' on parameters ' + str(learner.getparams()))

					trainset1 = trainset[1].reshape(trainset[1].shape[0], 1)
					XSplitter = np.array_split(trainset[0], 5)
					YSplitter = np.array_split(trainset1, 5)

					avgError = []
					nameParam.append(params)
					
					for k in range(k_fold):
						trainX1 = np.zeros((1000,9))
						trainX1 = XSplitter[k]

						trainY1 = np.zeros((1000,1))
						trainY1 = YSplitter[k]

						trainX0 = np.array([], dtype=np.int64).reshape(0,9)
						trainY0 = np.array([], dtype=np.int64).reshape(0,1)

						for j in range(k_fold):
							if j != k:
								trainX0 = np.vstack([trainX0, XSplitter[j]])
								trainY0 = np.vstack([trainY0, YSplitter[j]])

						# Train model
						learner.learn(trainX0, trainY0)
						# Test model
						predictions = learner.predict(trainX1)
						error = geterror(trainY1, predictions)
						print ('Error for ' + learnername + ' for k ' + str(k) +' as testset of trainings: ', end='')
						print(colored(str(error), 'red'))
						avgError.append(error)
#						CVError[learnername][p, k_fold, r] = error

						if k == 4:
							meanError = np.mean(avgError)
							print("our k-fold erros are: " + str(avgError))
							print("our mean error is: ", end = '')
							print(colored(str(meanError), 'red'))
							meanErrParam.append(meanError)
					
					if p == numparams - 1:
#						print(nameParam)
#						print("our mean Error for this parameter" + str(meanErrParam))
						index = np.argmin(meanErrParam)
						print("mean error is " + str(meanErrParam[index]) + " is for " + str(nameParam[index]))
						params = nameParam[index]
						
				
				print("parameters are " + str(params))		
				learner.reset(params)
				print ('Running learner = ' + learnername + ' on parameters ' + str(learner.getparams()))
				# Train model
				learner.learn(trainset[0], trainset[1])
				# Test model
				predictions = learner.predict(testset[0])
				error = geterror(testset[1], predictions)
				print ('Error for ' + learnername + ': ', end='')
				print(colored(str(error), 'red'))
				errors[learnername][p,r] = error


	if k_fold != 0 and k_foldStratified == 1:


		for r in range(numruns):
			trainset, testset = dtl.load_susy(trainsize,testsize)
			#trainset, testset = dtl.load_susy_complete(trainsize,testsize)
			#trainset, testset = dtl.load_census(trainsize,testsize)

			
			
			print(('Running on train={0} and test={1} samples for run {2}').format(trainset[0].shape[0], testset[0].shape[0],r))

			for learnername, learner in classalgs.items():

				meanErrParam = []
				nameParam = []
				params = ""
				for p in range(numparams):

					params = parameters[p]
					learner.reset(params)
					print ('Running learner = ' + learnername + ' on parameters ' + str(learner.getparams()))
	#				print(trainX0.shape)
	#				print(trainY0.shape)

					class0Index = []
					class1Index = []
					trainset1 = trainset[1].reshape(trainset[1].shape[0], 1)
					XSplitter = np.array_split(trainset[0], 5)
					for i in range(trainset[1].shape[0]):
						if trainset[1][i] == 1:
							class1Index.append(i)
						if trainset[1][i] == 0:
							class0Index.append(i)
#					print(len(class1Index))
#					print(len(class0Index))
					
					ratio = len(class1Index) / len(class0Index)
					
					YSplitter = np.array_split(trainset1, 5)
					numberOfClass0 = int((trainset[1].shape[0] / k_fold) / (ratio + 1))
					numberOfClass1 = int((trainset[1].shape[0] / 5) - numberOfClass0)
					print(numberOfClass0)
					print(class0Index[numberOfClass0] - class0Index[0])
					
#					print(numberOfClass0)
#					print(numberOfClass1)
					
					#initializing 2D empty list
					YSplitter = []
					XSplitter = []
					for i in range(k_fold):
						YSplitter.append([])
						YSplitter[i] = np.asarray(YSplitter[i])
						YSplitter[i] = YSplitter[i].reshape(0,1)
						XSplitter.append([])
						XSplitter[i] = np.asarray(XSplitter[i])
						XSplitter[i] = XSplitter[i].reshape(0,9)

					
					#put values in (k_fold - 1) lists because the remaining should go to the last class
					for i in range(k_fold - 1):
						for j in range(numberOfClass0):
							YSplitter[i] = np.vstack([YSplitter[i], trainset1[class0Index[j]]])
							XSplitter[i] = np.vstack([XSplitter[i], trainset[0][class0Index[j]]])
						for k in range(numberOfClass1):
							YSplitter[i] = np.vstack([YSplitter[i], trainset1[class1Index[k]]])
							XSplitter[i] = np.vstack([XSplitter[i], trainset[0][class1Index[k]]])
						class0Index[0:numberOfClass0] = []
						class1Index[0:numberOfClass1] = []
					
					for j in range(len(class0Index)):
						YSplitter[k_fold - 1] = np.vstack([YSplitter[i], trainset1[class0Index[j]]])
						XSplitter[k_fold - 1] = np.vstack([XSplitter[i], trainset[0][class0Index[j]]])					
					for j in range(len(class1Index)):
						YSplitter[k_fold - 1] = np.vstack([YSplitter[i], trainset1[class1Index[j]]])
						XSplitter[k_fold - 1] = np.vstack([XSplitter[i], trainset[0][class1Index[j]]])
						

					avgError = []
					nameParam.append(params)
					
					for k in range(k_fold):
						trainX1 = np.zeros((1000,9))
						trainX1 = XSplitter[k]

						trainY1 = np.zeros((1000,1))
						trainY1 = YSplitter[k]

						trainX0 = np.array([], dtype=np.int64).reshape(0,9)
						trainY0 = np.array([], dtype=np.int64).reshape(0,1)

						for j in range(k_fold):
							if j != k:
								trainX0 = np.vstack([trainX0, XSplitter[j]])
								trainY0 = np.vstack([trainY0, YSplitter[j]])

						# Train model
						learner.learn(trainX0, trainY0)
						# Test model
						predictions = learner.predict(trainX1)
						error = geterror(trainY1, predictions)
						print ('Error for ' + learnername + 'k ' + str(k) +': ' + str(error))
						avgError.append(error)
#						CVError[learnername][p, k_fold, r] = error

						if k == 4:
							meanError = np.mean(avgError)
							print(avgError)
							print(meanError)
							meanErrParam.append(meanError)
					
					if p == numparams - 1:
						print(nameParam)
						print(meanErrParam)
						index = np.argmin(meanErrParam)
						print("mean error is " + str(meanErrParam[index]) + " is for " + str(nameParam[index]))
						params = nameParam[index]
						
				
				print("parameters are " + str(params))		
				learner.reset(params)
				print ('Running learner = ' + learnername + ' on parameters ' + str(learner.getparams()))
				# Train model
				learner.learn(trainset[0], trainset[1])
				# Test model
				predictions = learner.predict(testset[0])
				error = geterror(testset[1], predictions)
				print ('Error for ' + learnername + ': ' + str(error))
				errors[learnername][p,r] = error

				
		
		
		
		
	else:	
		
		for r in range(numruns):
			trainset, testset = dtl.load_susy(trainsize,testsize)
			#trainset, testset = dtl.load_susy_complete(trainsize,testsize)
	#		trainset, testset = dtl.load_census(trainsize,testsize)

			print(('Running on train={0} and test={1} samples for run {2}').format(trainset[0].shape[0], testset[0].shape[0],r))

			nameParam = []
			meanErrParam = []
			
			for p in range(numparams):
				params = parameters[p]
				
				for learnername, learner in classalgs.items():
					# Reset learner for new parameters
					learner.reset(params)
					print ('Running learner = ' + learnername + ' on parameters ' + str(learner.getparams()))
					# Train model
					learner.learn(trainset[0], trainset[1])
					# Test model
					predictions = learner.predict(testset[0])
					error = geterror(testset[1], predictions)
					print ('Error for ' + learnername + ': ' + str(error))
					errors[learnername][p,r] = error


		for learnername, learner in classalgs.items():
			besterror = np.mean(errors[learnername][0,:])
			bestparams = 0
			for p in range(numparams):
				aveerror = np.mean(errors[learnername][p,:])
				if aveerror < besterror:
					besterror = aveerror
					bestparams = p

		# Extract best parameters
		learner.reset(parameters[bestparams])
		print ('Best parameters for ' + learnername + ': ' + str(learner.getparams()))
		print ('Average error for ' + learnername + ': ' + str(besterror) + ' +- ' + str(np.std(errors[learnername][bestparams,:])/math.sqrt(numruns)))
