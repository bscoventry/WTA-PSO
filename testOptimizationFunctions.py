"""
Created on Thur Jul 05 21:18:31 2018

@author: Brandon S Coventry. Weldon School of Biomedical Engineering Purdue University
Purpose: This file contains all test functions for evaluating WTAPSO
Revision History: N/A
Notes: Reference: Coventry et al., 2017 J. Comput Neurosci has all problem definitions
"""
import numpy as np              #Will use inf and other numpy functions
def Rosenbrock(inputvals):
    """ Multidimensional rosenbrock function. Must have n>2 values. Assumes numpy input array
    Global minimum of 0 at inputvals = [1,1,1,.....,1,1] """
    probDim = len(inputvals)
    if probDim < 2:
        raise ValueError('Input must have a length of greater or equal to 2')
    else:
        totVal = np.empty([probDim]) #Array to store values and eventually sum
        for ck in range(probDim-1):
            curval = 100.*(np.square(np.square(inputvals[ck])-inputvals[ck+1]))+np.square(inputvals[ck]-1)
            totVal[ck] = curval
        Rosenval = np.sum(totVal)
    return Rosenval

def Sphere(inputvals):
    """ Multidimensional Sphere function. Assumes numpy input array
    Global minimum of 0 at inputvals = [0,0,0,.....,0,0] """
    probDim = len(inputvals)
    totVal = np.empty([probDim])
    for bc in range(probDim-1):
        totVal[bc] = np.square(inputvals[bc])
    Sphereval = np.sum(totVal)
    return Sphereval
    
def Griewank(inputvals):
    """ Multidimensional Griewank function. Assumes numpy input array
    Global minimum of 0 at inputvals = [0,0,0,...,0,0,0]
    input values must be between -600,600 """ 
    probDim = len(inputvals)
    sumArray = np.empty([probDim])
    prodArray = np.empty([probDim])
    for ii in range(probDim):
        curterm1 = np.square(inputvals[ii])
        curterm2 = np.cos(inputvals[ii]/np.sqrt(ii+1))
        sumArray[ii] = curterm1
        prodArray[ii] = curterm2
    Grieval = 1+((1/4000.)*np.sum(sumArray))-np.prod(prodArray)
    return Grieval

def Rastrigin(inputvals):
    """ Multidimensional Rastrigin function. Assumes numpy input array
    Global minimum of 0 at inputvals = [0,0,0,...,0,0,0]
    input values must be between -5.12,5.12 """
    probDim = len(inputvals)
    A = 10.         #Constant defined in the problem
    sumArray = np.empty([probDim])
    for jj in range(probDim):
        curval = np.square(inputvals[jj])-A*np.cos(2*np.pi*inputvals[jj])
        sumArray[jj] = curval
    Rastval = A*probDim + np.sum(sumArray)
    return Rastval

def SchafferF2(inputvals):
    """ 2 Dimensional Schaffer F2 function. Assumes numpy input array of size 2
    Global minimum of 0 at inputvals = [0,0]
    input vals must be between -100 and 100 """
    testDim = len(inputvals)
    if testDim == 2:
        num = np.square(np.sin(np.square(inputvals[0])-np.square(inputvals[1]))) - 0.5
        den = 1. + 0.001*np.square(np.square(inputvals[0])+np.square(inputvals[1]))
        SchF2 = 0.5+(num/den)
    else:
        raise ValueError('Input dimension must be exactly 2')
    return SchF2
    
def SchafferF4(inputvals):
    """ 2 Dimensional Schaffer F4 function. Assumes numpy input array of size 2
    Global minimum of 0.292579 at inputvals = [0,1.253115]
    input vals must be between -100 and 100 """
    testDim = len(inputvals)
    if testDim == 2:
        num = np.cos(np.sin(np.abs(np.square(inputvals[0])-np.square(inputvals[1]))))**2-0.5
        den = 1. + 0.001*np.square(np.square(inputvals[0])+np.square(inputvals[1]))
        SchF4 = 0.5+(num/den)
    else:
        raise ValueError('Input dimension must be exactly 2')
    return SchF4

def Styblinski(inputvals):
    """ Multidimensional Styblinski-Tang function. Assumes numpy input array
    Global minimum of -39.16599*dimension at inputvals = [-2.903534,...,-2.903534]
    input values must be between -5,5 """
    probDim = len(inputvals)
    sumArray = np.empty([probDim])
    for ck in range(probDim):
        curval = (inputvals[ck]**4)-16*(inputvals[ck]**2)+5*inputvals[ck]
        sumArray[ck] = curval
    Styb = np.sum(sumArray)/2.
    return Styb
    
    
    