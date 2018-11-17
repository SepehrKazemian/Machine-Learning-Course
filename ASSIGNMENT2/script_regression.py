from __future__ import division  # floating point division
import csv
import random
import math
import numpy as np
import utilities as util
import matplotlib.pyplot as plt

import dataloader as dtl
import regressionalgorithms as algs

def l2err(prediction,ytest):
    """ l2 error (i.e., root-mean-squared-error) """
    return np.linalg.norm(np.subtract(prediction,ytest))

def l1err(prediction,ytest):
    """ l1 error """
    return np.linalg.norm(np.subtract(prediction,ytest),ord=1)

def l2err_squared(prediction,ytest):
    """ l2 error squared """
    return np.square(np.linalg.norm(np.subtract(prediction,ytest)))

def geterror(predictions, ytest):
    # I have changed this geterror by the permission of the TA
    return l2err(predictions,ytest) **2 / (2 * ytest.shape[0])


if __name__ == '__main__':
    trainsize = 1000
    testsize = 5000
	#number of runs should be larger than 1 to make it possible to calculate standard deviation for standard error
    numruns = 2

    regressionalgs = {#'Random': algs.Regressor(),
#                'Mean': algs.MeanPredictor(),
#                'FSLinearRegression5': algs.FSLinearRegression({'features': [1,2,3,4,5]}),
                'FSLinearRegression50': algs.FSLinearRegression({'features': range(385)}),
                'RidgeLinearRegression': algs.RidgeLinearRegression({'features': range(385)}),
		'LassoRegression' : algs.LassoRegression({'features': range(385)}),
		'StochasticGradientDescent' : algs.StochasticGradientDescent({'features': range(385)}),
		'BatchGradientDescent' : algs.BatchGradientDescent({'features': range(385)}),
		'RMSProp' : algs.RMSProp({'features': range(385)}),
		'AMSGrad' : algs.AMSGrad({'features': range(385)}),
             }
    numalgs = len(regressionalgs)

    # Enable the best parameter to be selected, to enable comparison
    # between algorithms with their best parameter settings
    parameters = (
        {'regwgt': 0.01},
        {'regwgt': 0.0},
        {'regwgt': 1.0},
                      )
    numparams = len(parameters)
    
    errors = {}
    epoch_error = {}
    sgdEpochErr = []
    rmsEpochErr = []
    amsEpochErr = []
    bgdEpochErr = []
    EpochErrSGD = []
    EpochErrBGD = []
    EpochErrAMS = []
    EpochErrRMS = []

    for i in range(numparams):
        tmp = []
        for j in range(numruns):
            tmp.append([])
		
    for learnername in regressionalgs:
        errors[learnername] = np.zeros((numparams,numruns))
        epoch_error[learnername] = np.zeros((numparams,numruns))
    		
    EpochErrSGD = [[0 for params in range(numparams)] for run in range(numruns)]
    EpochErrAMS = [[0 for params in range(numparams)] for run in range(numruns)]
    EpochErrRMS = [[0 for params in range(numparams)] for run in range(numruns)]
    TimeErrSGD = [[0 for params in range(numparams)] for run in range(numruns)]
    TimeErrAMS = [[0 for params in range(numparams)] for run in range(numruns)]
    TimeErrRMS = [[0 for params in range(numparams)] for run in range(numruns)]
    EpochErrBGD = [[0 for params in range(numparams)] for run in range(numruns)]
    TimeErrBGD = [[0 for params in range(numparams)] for run in range(numruns)]
    for r in range(numruns):
        trainset, testset = dtl.load_ctscan(trainsize,testsize)
        print(('Running on train={0} and test={1} samples for run {2}').format(trainset[0].shape[0], testset[0].shape[0],r))

        for p in range(numparams):
            params = parameters[p]
            for learnername, learner in regressionalgs.items():
                # Reset learner for new parameters
                learner.reset(params)
                print ('Running learner = ' + learnername + ' on parameters ' + str(learner.getparams()))
                # Train model
                learner.learn(trainset[0], trainset[1])
                # Test model
                predictions = learner.predict(testset[0])
                error = geterror(testset[1], predictions)
                if learnername == "StochasticGradientDescent":
                    tmp, timer = learner.epoch_error(testset[0])
                    for i in range(1000):
                        sgdEpochErr.append(geterror(tmp[i], testset[1]))

                    EpochErrSGD[r][p] = sgdEpochErr
                    TimeErrSGD[r][p] = timer
                    sgdEpochErr = []

                if learnername =="RMSProp":
                    tmp, timer = learner.epoch_error(testset[0])
                    for i in range(1000):
                        rmsEpochErr.append(geterror(tmp[i], testset[1]))
                    EpochErrRMS[r][p] = rmsEpochErr
                    TimeErrRMS[r][p] = timer
                    rmsEpochErr = []


                if learnername == "AMSGrad":
                    tmp, timer = learner.epoch_error(testset[0])
                    for i in range(1000):
                        amsEpochErr.append(geterror(tmp[i], testset[1]))
                    EpochErrAMS[r][p] = amsEpochErr
                    TimeErrAMS[r][p] = timer
                    amsEpochErr = []
                    
                if learnername == "BatchGradientDescent":
                    tmp, timer = learner.epoch_error(testset[0])
                    for i in range(len(timer)):
                        bgdEpochErr.append(geterror(tmp[i], testset[1]))
   
                    EpochErrBGD[r][p] = bgdEpochErr
                    TimeErrBGD[r][p] = timer
                    bgdEpochErr = []
                print ('Error for ' + learnername + ': ' + str(error))
                errors[learnername][p,r] = error
    arrCounterSGD = []
    arrCounterBGD = []
    arrCounterRMS = []
    arrCounterAMS = []
    a = 1
    for i in range(1000):
        arrCounterSGD.append(a)
        a += 1
    a = 1
    for i in range(len(EpochErrBGD[0][0])):
        arrCounterBGD.append(a)
        a += 1
	
    plt.figure(1)  
    plt.subplot(211)
    plt.title('SGD (top) vs BGD (bottom) in 1000 epochs')  
    plt.plot(arrCounterSGD[0:-1], EpochErrSGD[0][0][0:-1], color='darkblue', linewidth=0.5)
    plt.xlabel('Epochs')
    plt.ylabel('MSE')


    plt.subplot(212)
    plt.plot(arrCounterBGD[0:-1], EpochErrBGD[0][0][0:-1], color='red', linewidth=0.5)
    plt.xlabel('Epochs')
    plt.ylabel('MSE')

#    print("error bgd is: " + str(EpochErrBGD))
#    print("error sgd is: " + str(EpochErrSGD))
#    runs = range(1,numruns+1)
    if len(arrCounterBGD) < len(arrCounterSGD):
        EpochErrSGD[0][0] = EpochErrSGD[0][0][0:len(arrCounterBGD)]
        TimeErrSGD[0][0] = TimeErrSGD[0][0][0:len(arrCounterBGD)]

    lastTime = TimeErrBGD[0][0][-1]
    indexCatcher = 0
    for i in range(len(TimeErrSGD[0][0])):
        if TimeErrSGD[0][0][i] > lastTime:
               indexCatcher = i
               break
    TimeErrSGD[0][0] = TimeErrSGD[0][0][0:indexCatcher]
    EpochErrSGD[0][0] = EpochErrSGD[0][0][0:indexCatcher]
    

    plt.figure(2)

    plt.subplot(211)
    plt.title('SGD (top) vs BGD (bottom) in time')  
    plt.plot(TimeErrSGD[0][0][0:-2], EpochErrSGD[0][0][0:-2], color='black', linewidth=0.5)
    plt.xlabel('Time (ms)')
    plt.ylabel('MSE')

    plt.subplot(212)
    plt.plot(TimeErrBGD[0][0][0:-2], EpochErrBGD[0][0][0:-2], color='green', linewidth=0.5)
    plt.xlabel('Time (ms)')
    plt.ylabel('MSE')



    plt.figure(3)
    plt.subplot(211)
    plt.title('RMSProp (top) vs AMSGrad (bottom) in 1000 epochs')  
    plt.plot(arrCounterSGD, EpochErrRMS[0][0], color='black', linewidth=0.5)
    plt.xlabel('Epochs')
    plt.ylabel('MSE')

    plt.subplot(212)
    plt.plot(arrCounterSGD, EpochErrAMS[0][0], color='green', linewidth=0.5)
    plt.xlabel('Epochs')
    plt.ylabel('MSE')

    plt.show()




    for learnername in regressionalgs:
        besterror = np.mean(errors[learnername][0,:])
		#finding the standard deviation with numpy.std is one way to to do it
        std_err = np.std(errors[learnername][0, :], ddof=1)
		#finding the standard deviation by using utilities.py's implemented function. The result is the same
		#with numpy.std
        SDUtil = util.stdev(np.array(errors[learnername][0, :]))
		
        bestparams = 0
        for p in range(numparams):
            aveerror = np.mean(errors[learnername][p,:])
            if aveerror < besterror:
                std_err = np.std(errors[learnername][p, :], ddof=1)
                SDUtil = util.stdev(np.array(errors[learnername][0, :]))
                besterror = aveerror
                bestparams = p
        #By using the standard deviation function in module utilities, we computed the standard deviation
        #of errors for each rewgt for several runs of each available algorithm. get the minimum standard 
        #deviation error over all of the three regwgt parameters        

		#here we calculate standard error for each of the standard deviation that we have calculated (numpy and utilities)
        std_err = std_err/math.sqrt(numruns)
        SDUtil = SDUtil/math.sqrt(numruns)
        # Extract best parameters
        learner.reset(parameters[bestparams])
        #print ('Best parameters for ' + learnername + ': ' + str(learner.getparams()))
        print ('Average error for ' + learnername + ': ' + str(besterror))
        print("Standard Error for " + learnername + " with numpy.std is: " + str(std_err))
        print("Standard Error for " + learnername + " with utilities.stdev is: " + str(SDUtil))
