# -*- coding: utf-8 -*-
"""
Created on Mon Jul 09 15:14:32 2018

@author: Brandon S Coventry. Weldon School of Biomedical Engineering Purdue University
Purpose: This is the Control and run file which operates the swarm. 
Included files: AgentClass - Holds the class functions for WTAPSO
                testOptimizationFunctions - Mathematical definitions of test functions used to test WTAPSO 
                WTAPSO_HelperFunctions - File holding helper functions to assist in swarm creation and running
Revision History: 5/12/20. Added GPU support via cupy.
Notes: Reference: Coventry et al., 2017 J. Comput Neurosci
"""
""" Define all necessary includes for WTAPSO """
#from neuron import *                   #Only needed if running Neuron optimizations
#from nrn import *                      #Only needed if running Neuron optimizations
from testOptimizationFunctions import * #Import all for now. Can comment out if more memory is needed
#from AgentClass import *                #Import all necessary classes
from WTAPSO_HelperFunctions import *    #Import all the swarm helper functions
from WTAPSO import WTAPSO               #Swarm functionality
import cupy as np                      #For scientific programming operations. cupy used for GPU support of numpy functions. Change to numpy if CPU only
import random as rnd
import scipy.io
import matplotlib.pyplot as mpl         #For generating plots
import pdb                              #For code diagnostics
"""Set swarm variables"""
phi1 = 2.05                             #Phi1,phi2, and chi control swarming behaviour, trading off between exploration and seeking trends towards good results
phi2 = 2.05
chi =  0.7298
problemDimension = 2             #Dimensionality of problem under optimization
numIter = 100000                         #This version of the swarm has a fixed run iteration, will update for complex stopping criterion
costfunc = Rosenbrock#lambda inputvec: Rosenbrock(inputvec)     #This is the function you wish to optimize. Takes an inputvector of values. 
softConstraintsLow = np.inf*np.ones([1,problemDimension])               #Option for soft constraint of features. Set to inf if it is to be ignored.
softConstraintsHigh = np.inf*np.ones([1,problemDimension])
softConstraints = [softConstraintsLow,softConstraintsHigh]
randstart = [-500,500]             #Choose this to start large to sample space. The swarm will draw random values between these two to place an agent in the solution space
""" Release the swarm! """
[bestagent,errorvec,swarm] = WTAPSO(problemDimension,numIter,Rosenbrock,softConstraints,randstart,phi1,phi2,chi)
print(errorvec)
""" Results of Optimization """
returnOptimizationResults(bestagent)
pdb.set_trace()
# """ Setup problem definition here """
# problemDim = 2                      #Dimension of the problem you'd like to solve
# lowconst = -100#-10000000                     #Lower bound soft constraint for PSO
# highconst = 100#10000000                    #Upper bound for soft constraint
# costfunc = lambda inputvec: Rosenbrock(inputvec)     #This is the function you wish to optimize. Takes an inputvector of values. 
# numiterations = 1000                   #Number of iterations before artificial stopping
# """ Setup swarm here """
# swarmsize = 21                          #For the moment, keep this fixed.
# chi = 0.7298                            #Constant for growth constraint
# phi1 = 2.05
# phi2 = 2.05
# phi = phi1+phi2
# swarm = [Particle(np.random.uniform(lowconst,highconst,[problemDim]),np.zeros(problemDim),chi,phi1,phi2,np.inf) for ck in range(swarmsize)]   #Generate the swarm!!!
# errorvec = np.ones(swarmsize)*np.inf               #Often helpful to keep track of fitness in an outside variable
# """ Initial run """
# for ii in range(swarmsize):
#     curagent = swarm[ii]                        #Grab agent in the swarm
#     curinputvals = curagent.returnPosition()      #Grab the inputs to the cost function
#     curfitness = costfunc(curinputvals)         #Calculate a new fitness
#     swarm[ii].setFitness(curfitness)            #Set fitness for each agent
#     swarm[ii].setBestPosition(curagent.returnPosition())       #At the first run, best position is the current position
#     errorvec[ii] = swarm[ii].returnFitness()
# """ Create Swarm """
# minerr = np.argmin(errorvec)                    #Grab the position of the best performing agent.
# bestagent = swarm[minerr]
# nummem = 5                                      #Number of members in the local networks. Fix at 5 for the moment.
# matrixdim = [nummem,4]                          #4 is fixed for the moment. Will update later
# [N,sizes,neisize,errind] = genSwarmNetwork(swarmsize,nummem,matrixdim,minerr)     #Generate swarm matrix
# row = sizes[0]                                  #Grab matrix dimensions for further processing
# col = sizes[1]
# #Okay, now generate error vectors for the neighbors and neighborhood leaders
# neibest = np.zeros(int(neisize))                    #Vector containing fitness for the best performing neighbors in the network to act as local leaders
# for kk in range(col):                 #Iterate through neighborhoods, which are columns of the swarm matrix N
#     curcol = kk
#     agentvals = N[:,kk]
#     errv = np.zeros(len(agentvals))
#     for ll in range(len(agentvals)):
#         curagent = swarm[int(agentvals[ll]-1)]
#         errv[ll] = curagent.returnFitness()
#     minern = np.where(errv==np.min(errv))
#     neibest[kk] = agentvals[minern[0][0]]
# agenterrorbest = swarm[minerr]
# agenterrbest = swarm[minerr].returnFitness()
# """ Update the swarm! """
# for yy in range(numiterations):         #Run for N interations
#     swarmup = swarm                   #Create a copy of the swarm to update
#     swarm = updateSwarm(swarmup,errorvec,bestagent,neibest,row,col,N,minerr) #Update the swarm!! Return updated swarm to swarm variable.
#     #Here we iterate through as before and recalculate fitness values
#     for pp in range(swarmsize):
#         agent = swarm[pp]
#         curinputvals = agent.returnPosition()      #Grab the inputs to the cost function
#         curfitness = costfunc(curinputvals)         #Calculate a new fitness
#         print(curfitness)
#         swarm[pp].setFitness(curfitness)            #Set fitness for each agent
#         errorvec[pp] = swarm[pp].returnFitness()
#         [N,sizes,neisize,errind]=genSwarmNetwork(swarmsize,nummem,matrixdim,minerr)
#         [N,sizes,neisize,agenterrorbest,agenterrbest,minerr] = updateSwarmNetwork(swarm,N,sizes,neisize,minerr,swarmsize,errorvec,errind)
# returnOptimizationResults(swarm,minerr,bestagent)
# print(errorvec)
    