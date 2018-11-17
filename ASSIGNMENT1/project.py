from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import random
import sys

def plot_gaussian(x,y,z,filename=None):
    """
    Plot the multivariate Gaussian

    If filename is not given, then the figure is not saved.
    """
    # Note: there was no need to make this into a separate function
    # however, it lets you see how to define functions within the
    # main file, and makes it easier to comment out plotting if
    # you want to experiment with many parameter changes without
    # generating many, many graphs
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x,y,z, c="red",marker="s")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    minlim, maxlim = -3, 3
    ax.set_xlim(minlim,maxlim)
    ax.set_ylim(minlim,maxlim)
    ax.set_zlim(minlim,maxlim)
    if filename is not None:
        fig.savefig("scatter" + str(dim) + "_n" + str(numsamples) + ".png")
    plt.show()


if __name__ == '__main__':

    # default
    dim = input("please enter the number of dimensions: ")
    numsamples = 1000
    
    print("Running with dim = " + str(dim), \
        " and numsamples = " + str(numsamples))
        
		
	sigma = [[0] * dim] * dim
	for i in range(dim):
		sigma[i][i] = 1
	
	
/***	
    # Generate data from (multivariate) Gaussian
    if dim == 1:
        # mean and standard deviation in one dimension
        mu = 0
        sigma = 10
        x = np.random.normal(mu, sigma, numsamples)        
        y = np.zeros(numsamples,)
        z = np.zeros(numsamples,)
    elif dim == 2:
        # mean and standard deviation in two dimension
        mu = [0,0]
        sigma = [[1,0],[0,1]]
        x,y = np.random.multivariate_normal(mu, sigma, numsamples).T    
        z = np.zeros(numsamples,)
    else:
        # mean and standard deviation in three dimension
        mu = [0,0,0]
        #sigma = [[1,0,1],[0,1,0],[1,0,1]]
        sigma = [[1,0,0],[0,1,0],[0,0,1]]
        #sigma = [[1,-0.5,-0.5],[-0.5,0.5,-0.5],[-0.5,-0.5,0.5]]
        x,y,z = np.random.multivariate_normal(mu, sigma, numsamples).T
        
    # Get the current estimate of the mean
    print(np.mean(x))
    
    # Print all in 3d space, but just project to 2d or 1d
    #plot_gaussian(x,y,z,"scatter" + str(dim) + "_n" + str(numsamples) + ".png")
    plot_gaussian(x,y,z)
***/