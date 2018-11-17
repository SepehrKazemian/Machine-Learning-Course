from __future__ import division  # floating point division
import math
import numpy as np


def mean(numbers):
    return sum(numbers)/float(len(numbers))
 
def stdev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
    return math.sqrt(variance)
    
def sigmoid(xvec):
    """ Compute the sigmoid function """
    # Cap -xvec, to avoid overflow
    # Undeflow is okay, since it get set to zero
    xvec[xvec < -100] = -100

    vecsig = 1.0 / (1.0 + np.exp(np.negative(xvec)))
 
    return vecsig

def dsigmoid(xvec):
    """ Gradient of standard sigmoid 1/(1+e^-x) """
    vecsig = sigmoid(xvec)
    return vecsig * (1 - vecsig)

def l2(vec):
    """ Squared l2 norm on a vector """
    return np.square(np.linalg.norm(vec))

def dl2(vec):
    """ Gradient of squared l2 norm on a vector """
    return vec

def l1(vec):
    """ l1 norm on a vector """
    return np.linalg.norm(vec, ord=1)

def threshold_probs(probs):
    """ Converts probabilities to hard classification """
    classes = np.ones(len(probs),)
    classes[probs < 0.5] = 0
    return classes
                          

def logsumexp(a):
    """
    Compute the log of the sum of exponentials of input elements.
    Modified scipys logsumpexp implemenation for this specific situation
    """

    awithzero = np.hstack((a, np.zeros((len(a),1))))
    maxvals = np.amax(awithzero, axis=1)
    aminusmax = np.exp((awithzero.transpose() - maxvals).transpose())

    # suppress warnings about log of zero
    with np.errstate(divide='ignore'):
        out = np.log(np.sum(aminusmax, axis=1))

    out = np.add(out,maxvals)

    return out

def update_dictionary_items(dict1, dict2):
    """ Replace any common dictionary items in dict1 with the values in dict2 
    There are more complicated and efficient ways to perform this task,
    but we will always have small dictionaries, so for our use case, this simple
    implementation is acceptable.
    """
    for k in dict1:
        if k in dict2:
            dict1[k]=dict2[k]
