from __future__ import division  # floating point division
import numpy as np
import math
#from sklearn.linear_model import Ridge
import utilities as utils
import script_regression as scrReg
from random import shuffle
import datetime
import pytz


class Regressor:
    """
    Generic regression interface; returns random regressor
    Random regressor randomly selects w from a Gaussian distribution
    """

    def __init__( self, parameters={} ):
        """ Params can contain any useful parameters for the algorithm """
        self.params = {}
        self.reset(parameters)

    def reset(self, parameters):
        """ Reset learner """
        self.weights = None
        self.resetparams(parameters)

    def resetparams(self, parameters):
        """ Can pass parameters to reset with new parameters """
        self.weights = None
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
        self.weights = np.random.rand(Xtrain.shape[1])

    def predict(self, Xtest):
        """ Most regressors return a dot product for the prediction """
        ytest = np.dot(Xtest, self.weights)
        return ytest

class RangePredictor(Regressor):
    """
    Random predictor randomly selects value between max and min in training set.
    """

    def __init__( self, parameters={} ):
        """ Params can contain any useful parameters for the algorithm """
        self.params = {}
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        self.min = 0
        self.max = 1
        
    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        self.min = np.amin(ytrain)
        self.max = np.amax(ytrain)

    def predict(self, Xtest):
        ytest = np.random.rand(Xtest.shape[0])*(self.max-self.min) + self.min
        return ytest

class MeanPredictor(Regressor):
    """
    Returns the average target value observed; a reasonable baseline
    """
    def __init__( self, parameters={} ):
        self.params = {}
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        self.mean = None
        
    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        self.mean = np.mean(ytrain)

    def predict(self, Xtest):
        return np.ones((Xtest.shape[0],))*self.mean


class FSLinearRegression(Regressor):
    """
    Linear Regression with feature selection, and ridge regularization
    """
    def __init__( self, parameters={} ):
        self.params = {'regwgt': 0.01, 'features': [1,2,3,4,5]}
        self.reset(parameters)

    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        # Dividing by numsamples before adding ridge regularization
        # to make the regularization parameter not dependent on numsamples
        numsamples = Xtrain.shape[0]
        Xless = Xtrain[:,self.params['features']]
		# I have changed np.linalg.inv to the np.linalg.pinv to get the pseudo inverse instead of actual inverse (Q2-a)
        self.weights = np.dot(np.dot(np.linalg.pinv(np.dot(Xless.T,Xless)/numsamples), Xless.T),ytrain)/numsamples

    def predict(self, Xtest):
        Xless = Xtest[:,self.params['features']]
        ytest = np.dot(Xless, self.weights)
        return ytest

class RidgeLinearRegression(Regressor):
    def __init__( self, parameters={} ):
        # Default parameters, any of which can be overwritten by values passed to params
        self.params = {'regwgt': 0.01, 'features': [1,2,3,4,5]}
        self.reset(parameters)
        self.clf = ""
		
    def learn(self, Xtrain, ytrain):
        numsamples = Xtrain.shape[0]
        Xless = Xtrain[:,self.params['features']]
		
        #Now we add this line to take the weights based on the formula
        self.weights = np.dot(np.dot(np.linalg.pinv((np.dot(Xless.T,Xless)+ self.params['regwgt'] * np.identity(np.dot(Xless.T, Xless).shape[0]))/numsamples), Xless.T),ytrain)/numsamples
        
		#We use another implemented class (sklearn.linear_model.Ridge) in python to test our code with
		#the result was the same
#        self.clf = Ridge(alpha = self.params['regwgt'])
#        self.clf.fit(Xtrain, ytrain)
		
    def predict(self, Xtest):
        Xless = Xtest[:, self.params['features']]	
#        pred = self.clf.predict(Xless)
#        print(pred)
        ytest = np.dot(Xless, self.weights)
#        print(ytest)
        return ytest
		
class LassoRegression(Regressor):
    
    def __init__ (self, parameters={}):
        self.params = {'regwgt': 0.5, 'features': [1,2,3,4,5]}
        self.reset(parameters)
		
    def proximity(self, eta, lam, wght):
        for i in range(wght.shape[0]):
            if wght[i]> eta * self.params['regwgt']:
                self.weights[i] = wght[i]-eta*self.params['regwgt']
				
            elif abs(wght[i])<= eta * self.params['regwgt']:
                self.weights[i] = 0
            
            elif wght[i] < eta*self.params['regwgt']:
                self.weights[i] = wght[i] + eta*self.params['regwgt']
				
    def learn(self, Xtrain, ytrain):
        maxIterCounter = 0
        numsamples = Xtrain.shape[0]
        maxIter = 10e4
        featureNum = Xtrain.shape[1]
        self.weights = np.zeros(featureNum)
        tolerance = 0.0001
        Xless = Xtrain[:, self.params['features']]
        XX = np.dot(Xless.T,Xless)/numsamples
        Xy = np.dot(Xless.T,ytrain)/numsamples
        err = float('inf') # a very large number
        eta = 1/(2*np.linalg.norm(XX))
        errRange = (np.linalg.norm(np.subtract(np.dot(Xless, self.weights),ytrain), ord = None)/numsamples * 1/2) # (1/(2n)) * l2(Xw-y) + regularization
#        errRange = np.dot((np.dot(Xless, self.weights) - ytrain).T,(np.dot(Xless, self.weights) - ytrain)) + + np.multiply(self.params['regwgt'], np.linalg.norm(self.weights, ord=1))

#        print("error tolreance is 0.0001")
        while abs(err - errRange) > tolerance and maxIterCounter < maxIter:
            err = errRange
            self.proximity(eta, self.params['regwgt'], self.weights - eta * np.dot(XX, self.weights) + eta * Xy)
            maxIterCounter += 1
            errRange = (np.linalg.norm(np.subtract(np.dot(Xless, self.weights),ytrain), ord = None)/numsamples * 1/2)
#            print("c(w) is: " + str(errRange))
#            print("error difference is: " + str(err - errRange))
            
			
			

    def predict(self, Xtest):
        Xless = Xtest[:, self.params['features']]	
        ytest = np.dot(Xless, self.weights)
        return ytest
		

class StochasticGradientDescent(Regressor):
    
    def __init__ (self, parameters={}):
        self.params = {'features': [1, 2, 3, 4, 5] ,'regwgt': 0.5}
        self.reset(parameters)
        self.weightArr = []
        self.weights = ""

    def learn(self, Xtrain, ytrain):
        timeIs = int(datetime.datetime.now(tz=pytz.utc).timestamp() * 1000)
        numsamples = Xtrain.shape[0]
        Xless = Xtrain[:, self.params['features']]
        featuresNum = Xtrain[:, self.params['features']].shape[1]
        self.weights = np.random.rand(featuresNum)
        numSampleArr = []
        shufflingList = list(range(numsamples))
        self.weightArr = [[0 for weight in range(featuresNum)] for epoch in range(1000)]
#        print(len(self.weightArr))
#        print(len(self.weightArr[0]))
#        print(len(self.weights))

        self.weightTimeArr = [[0 for weight in range(featuresNum)] for epoch in range(1000)]
        self.concat = []
        

        for i in range(1000):
            eta = 0.01/(i+1)
            shuffle(shufflingList)
            for j in shufflingList:
 #               g = np.dot(Xless[j,:].T, (np.subtract(np.dot(Xless[j,:],self.weights), ytrain[j])))
                g = np.dot((np.subtract(np.dot(Xless[j,:].T,self.weights), ytrain[j])), Xless[j,:] )
                self.weights = np.subtract(self.weights, eta * g)			
            self.weightArr[i] = self.weights
            timeNow = int(datetime.datetime.now(tz=pytz.utc).timestamp() * 1000)
            timeDiff = timeNow - timeIs
            self.concat.append(timeDiff)
#            print(self.weights)
#        print(self.weightArr)

#        print(self.weights)
	
    def predict(self, Xtest):
        Xless = Xtest[:, self.params['features']]
        ytest = np.dot(Xless, self.weights)
        return ytest

    def epoch_error(self, Xtest):
        epochErrArr = []
        for i in range(1000):
            epochErrArr.append(np.dot(Xtest, self.weightArr[i]))
        return epochErrArr, self.concat
				

class BatchGradientDescent(Regressor):
    

    def __init__ (self, parameters={}):
        self.params = {'features': [1, 2, 3, 4, 5] ,'regwgt': 0.5}
        self.reset(parameters)
        self.weightArr = []
		
    def reset(self, parameters):
        self.resetparams(parameters)
        self.weights = np.random.rand(len(self.params['features']))

    def learn(self, Xtrain, ytrain):
        """ using the algorithm 2 in the book """
        timeIs = int(datetime.datetime.now(tz=pytz.utc).timestamp() * 1000)
        numsamples = Xtrain.shape[0]
        featuresNum = Xtrain.shape[1]
        eta_max = 1
        self.weights = np.random.rand(featuresNum)
        err = float('inf') # a very large number
        tolerance = 0.00001
        Xless = Xtrain[:, self.params['features']]
        errRange = scrReg.geterror(np.dot(Xless, self.weights), ytrain)
        maxIterCounter = 0
        self.weightArr = [[0 for weight in range(featuresNum)] for epoch in range(1000)]
        self.weightTimeArr = [[0 for weight in range(featuresNum)] for epoch in range(1000)]
        self.concat = []
        while abs(errRange - err) > tolerance :
            taw = 0.7			
            err = errRange
            g = np.divide(np.dot(Xless.T,np.subtract(np.dot(Xless, self.weights),ytrain)), numsamples)
            w = self.weights

 #           self.epoch_weight.append(self.weights)
            self.lineSearch(self.weights, errRange, g, Xless, ytrain, numsamples)
            errRange = scrReg.geterror(np.dot(Xless, self.weights), ytrain)
            if maxIterCounter < 1000:
                    self.weightArr[maxIterCounter] = self.weights
                    timeNow = int(datetime.datetime.now(tz=pytz.utc).timestamp() * 1000)
                    timeDiff = timeNow - timeIs
                    self.concat.append(timeDiff)
                    maxIterCounter += 1

        if maxIterCounter < 1000:
             del self.weightArr[maxIterCounter - 1: -1]
#        print("length of concat: " + str(len(self.concat)))
#        print("length of concat: " + str(len(self.weightArr)))
			
			
    def lineSearch(self, w, cw, g, Xless, ytrain, numsamples):
            taw = 0.7
            linesearch_tolerance = 0.000001
            eta = 1
            obj = cw
            counter = 0
            while (counter < 100):
                counter += 1
                newWeight = w - eta * g				
                newCw = scrReg.geterror(np.dot(Xless, newWeight), ytrain)
                if newCw < (obj - linesearch_tolerance):
                    self.weights = newWeight
                    break
                eta = taw * eta
            if counter == 100:
                print("Could not improve the solution")

			
    def predict(self, Xtest):
        ytest = np.dot(Xtest, self.weights)
        return ytest
		
    def epoch_error(self, Xtest):
        epochErrArr = []
        for i in range(len(self.weightArr)):
            epochErrArr.append(np.dot(Xtest, self.weightArr[i]))
        return epochErrArr, self.concat



class RMSProp(Regressor):
    
    def __init__ (self, parameters={}):
        self.params = {'features': [1, 2, 3, 4, 5] ,'regwgt': 0.5}
        self.reset(parameters)
        self.weightArr = []
        self.weights = ""

    def learn(self, Xtrain, ytrain):
        timeIs = int(datetime.datetime.now(tz=pytz.utc).timestamp() * 1000)
        numsamples = Xtrain.shape[0]
        Xless = Xtrain[:, self.params['features']]
        featuresNum = Xtrain[:, self.params['features']].shape[1]
        self.weightArr = [[0 for weight in range(featuresNum)] for epoch in range(1000)]
        self.weights = np.random.rand(featuresNum)
        numSampleArr = []
        shufflingList = list(range(numsamples))
        self.concat = []
        e = 1e-3
        p = 0.9

        for i in range(1000):
            v_tmp = 0
            eta = 0.001
            v = np.zeros(featuresNum)
            shuffle(shufflingList)
            for j in shufflingList:
                g = np.dot((np.subtract(np.dot(Xless[j,:].T,self.weights), ytrain[j])), Xless[j,:] )
                v = p * v + (1 - p) * (g**2)
                self.weights = self.weights - (eta / np.sqrt(v+ e) * g )
            self.weightArr[i] = self.weights
            timeNow = int(datetime.datetime.now(tz=pytz.utc).timestamp() * 1000)
            timeDiff = timeNow - timeIs
            self.concat.append(timeDiff)

            self.weightArr.append(self.weights)
#            print(self.weights)


    def predict(self, Xtest):
        Xless = Xtest[:, self.params['features']]
        ytest = np.dot(Xless, self.weights)
        return ytest

    def epoch_error(self, Xtest):
        epochErrArr = []
        for i in range(len(self.weightArr)):
            epochErrArr.append(np.dot(Xtest, self.weightArr[i]))
        return epochErrArr, self.concat

class AMSGrad(Regressor):
    
    def __init__ (self, parameters={}):
        self.params = {'features': [1, 2, 3, 4, 5] ,'regwgt': 0.5}
        self.reset(parameters)
        self.weightArr = []
        self.weights = ""

    def learn(self, Xtrain, ytrain):
        timeIs = int(datetime.datetime.now(tz=pytz.utc).timestamp() * 1000)
        numsamples = Xtrain.shape[0]
        Xless = Xtrain[:, self.params['features']]
        featuresNum = Xtrain[:, self.params['features']].shape[1]
        self.weightArr = [[0 for weight in range(featuresNum)] for epoch in range(1000)]
        self.weights = np.random.rand(featuresNum)
        numSampleArr = []
        shufflingList = list(range(numsamples))
        e = 1e-3
        p = 0.999
        self.concat = []

        for i in range(1000):
            self.m = 0
            self.v = 0
            v_hat = self.v
            eta = 0.001
            shuffle(shufflingList)
            for j in shufflingList:
                g = np.dot((np.subtract(np.dot(Xless[j,:].T,self.weights), ytrain[j])), Xless[j,:] )
                self.v = 0.99 * self.v + 0.01 * (g**2)
                self.m = 0.9 * self.m + 0.1 * g
  #              print(self.v)
 #               print(self.m)
                v_hat = np.maximum(v_hat, self.v)
                self.weights = self.weights - eta / (np.sqrt(v_hat) + e) * self.m
            self.weightArr[i] = self.weights
            timeNow = int(datetime.datetime.now(tz=pytz.utc).timestamp() * 1000)
            timeDiff = timeNow - timeIs
            self.concat.append(timeDiff)
				
            self.weightArr.append(self.weights)

    def predict(self, Xtest):
        Xless = Xtest[:, self.params['features']]
        ytest = np.dot(Xless, self.weights)
        return ytest

    def epoch_error(self, Xtest):
        epochErrArr = []
        for i in range(len(self.weightArr)):
            epochErrArr.append(np.dot(Xtest, self.weightArr[i]))
        return epochErrArr, self.concat
