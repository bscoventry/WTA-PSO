# -*- coding: utf-8 -*-
"""
Created on Mon Jul 09 21:33:39 2018

@author: Brandon S Coventry. Weldon School of Biomedical Engineering Purdue University
Purpose: This file holds helper functions which are used to assist in swarm creation and running
Revision History: N/A
Notes: Reference: Coventry et al., 2017 J. Comput Neurosci
"""
import numpy as np
from AgentClass import *    
def genSwarmNetwork(swarmsize,neisize,nummem,matrixdim,minerr):
    """ This function will generate the swarm network for use in running WTAPSO. Will do so by iterating through
    the agents performance and setting up a matrix to dictate each agents position in the swarm. """
    k = 0                             #Variable k is used in modulo to create network
    N = np.zeros([matrixdim[0],matrixdim[1]])        #N Holds our network
    for jj in range(swarmsize):           #Iterate through the agents and place them in the network
        if k <= nummem-1:
            N[k,int(np.mod(jj,neisize))] = jj
        else:
            k = 0                         #Reset mod factor
            N[k,int(np.mod(jj,neisize))] = jj
        k = k + 1
    #If agent 21 isn't a leader, then all agents are placed except 21. We swap it in for the global leader
    errind = np.where(N==minerr) #Find where best agent is in the matrix
    if minerr != swarmsize-1:
        N[errind[0],errind[1]] = swarmsize
    sizes = N.shape
    return [N,sizes,errind]
def updateNeighborErrors(swarm,N,neisize,neibest,minerr,swarmsize,errorvec,errind,row,col):
    """ This function is used to update the N matrix (matrix containing neighborhood agent information) in the update step
    of the WTAPSO. To be used after the first run of the swarm and all consequent update steps. """
    #errind = np.where(N == minerr)
    #if minerr != swarmsize-1:
        #N[errind[0],errind[1]] = swarmsize
    #sizes = N.shape
    #N[errind[0],errind[1]] = swarmsize             # Replace the best (it will have its own holder) with agent 21
    #sizes = N.shape
    #row = sizes[0]                              #Get the matrix dimensions
    #col = sizes[1]
    #neibest = np.zeros(int(neisize))
    for kk in range(col):                     #Iterate through neighborhoods, which are columns of N
        agentvals = N[:,kk]
        errv = np.zeros(len(agentvals))
        for ll in range(len(agentvals)):
            curagent = swarm[int(agentvals[ll]-1)]
            errv[ll] = curagent.returnFitness()
        minern = np.where(errv==np.min(errv))
        neibest[kk] = agentvals[minern[0][0]]
    #minerr = np.argmin(errorvec)
    agenterrorbest = swarm[minerr]
    agenterrbest = agenterrorbest.returnFitness
    return [agenterrorbest,agenterrbest,neibest]

def updateSwarm(swarm,errorvec,bestagent,neibest,row,col,N,minerr,constraints):
    """ This function updates the swarm. Loops through and finds global and local leaders
    and reorganizes the swarm based on these values."""
    swarmup = swarm                     #Create an identical copy for update.
    neibesterrors = errorvec[neibest.astype(int)-1]       #Grab the neighborhood leader values
    neibesterr = np.min(neibesterrors)
    if np.size(neibesterr)>1:
        neibesterr = neibesterr[1]           #Check for duplicates
    errneibestindex = np.where(errorvec==neibesterr)
    bestneileader = swarm[errneibestindex[0][0]]
    bestagent.updateAgent(bestagent.returnBestPosition(),bestneileader.returnBestPosition(),constraints)  #Class function to update. See update description for how best, neibest, and regular agents are updated
    swarmup[minerr] = bestagent
    """Now update neighborhood leaders"""
    for nn in range(col):
        updateneibest = neibest[nn]
        update = swarmup[int(updateneibest)-1]         #Grab the neighborhood leader for each subswarm
        neighbor = N[:,nn]
        neierrvec = errorvec[neighbor.astype(int)-1]        #Vector containing errors of local leader agents
        neighbesterr = np.min(neierrvec)      #Now grab the error of the best neighborhood agent, use to grab best
        if np.size(neighbesterr)>1:
            neighbesterr = neibesterr[0]      #If multiple have same error value, just choose 1
        bestneigh = np.where(errorvec==neighbesterr)           #Find the location of the best neighbor in the swam
        updateneigh = swarm[bestneigh[0][0]-1]       #Now that we have found the best neighbor, we grab it from the swarm
        update.updateAgent(bestagent.returnBestPosition(),updateneigh.returnBestPosition(),constraints)
        swarmup[int(updateneibest)-1]=update
    """Now update the Neighborhood values"""
    for mm in range(col):               #We again poll the neighborhood
        currentneigh = N[:,mm]
        currentneiwhere = np.where(currentneigh!=neibest[mm])             #Search through the local neighborhood, ignoring the leader, we've already updated them
        currentneigh = currentneigh[currentneiwhere]            
        upneibest = swarm[int(neibest[mm])-1]       ##Since this agent is defined as a neighborhood leader, it will default be a part of the individual neighbor update
        for ii in range(len(currentneigh)):           #Update individual neighbors
            upvar = swarm[int(currentneigh[ii])-1]        #Grab the agent from the swarm
            neierrvecindwhere = np.where(currentneigh!=currentneigh[ii])
            neierrvecind = currentneigh[neierrvecindwhere]
            neierrvec = errorvec[(neierrvecind.astype(int))-1]        #Grab agent who is performing best
            neierrvec = np.min(neierrvec)
            if np.size(neierrvec) > 1:
                neierrvec = neierrvec[0]              #If multiple occur, just choose one.
            bnei = np.where(errorvec==neierrvec)
            upN = swarm[bnei[0][0]]            #Pull agent from swarm
            upvar.updateAgent(upneibest.returnBestPosition(),upN.returnBestPosition(),constraints)
            swarmup[int(currentneigh[ii])-1] = upvar
    return [swarmup,minerr]
def updateSwarmNetwork(swarm,swarmsize,neisize,nummem,matrixdim,minerr,N):
    k = 0                      #Starting point for neighborhood generation
    neisize = int(neisize)
    for jj in range(swarmsize):
        if k <= nummem-1:
            N[k,np.mod(jj,neisize)]= jj
        else:
            k = 0
            N[k,np.mod(jj,neisize)] = jj
        k = k+1
    #If agent 21 isn't a leader, then all agents are placed except 21. We need to swap 21 for global leader, which sits outside this network structure
    errind = np.where(N==minerr)               #find where best agent is in the network
    N[errind[0],errind[1]] = swarmsize         #Replace the best (it will have its own holder) with agent 21 and reorganize
    sizes = np.shape(N)                   #Get matrix dimensions
    row = sizes[0]
    col = sizes[1]       
    neibest = np.zeros(neisize)
    for kk in range(col):
        curcol = kk
        agentvals = N[:,kk]
        errv = np.zeros(len(agentvals))
        for rr in range(len(agentvals)):
            curagent = swarm[int(agentvals[rr])-1]
            errv[rr] = curagent.returnFitness()
        minern = np.where(errv==np.min(errv))
        agenterrorbest = swarm[minerr]
        agenterrbest = agenterrorbest.returnFitness
    return [N,agenterrorbest,agenterrbest]
def returnOptimizationResults(bestagent):
    print('Found minima at: '+ str(bestagent.returnBestPosition())+'With a Fitness of:' + str(bestagent.returnFitness()))

