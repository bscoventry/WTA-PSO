"""
Created on Mon Nov 4 09:40 2019

@author: Brandon S Coventry. Weldon School of Biomedical  Purdue University
Purpose: This is the control function for calling WTAPSO optimization
Included files: AgentClass - Holds the class functions for WTAPSO
                testOptimizationFunctions - Mathematical definitions of test functions used to test WTAPSO 
                WTAPSO_HelperFunctions - File holding helper functions to assist in swarm creation and running
Revision History: 05/12/20 GPU support added for calculations in numpy via cuPy module. 
Notes: Reference: Coventry et al., 2017 J. Comput Neurosci
"""
import cupy as np                 #GPU availability. Replace cupy with numpy for CPU support.
import pdb
from AgentClass import Particle
from WTAPSO_HelperFunctions import genSwarmNetwork,updateSwarmNetwork,updateNeighborErrors,updateSwarm    #Import all the swarm helper functions
class WTAPSO(object):
    """ This is the Winner-Take-All Particle Swarm optimization class. Takes in the cost function on input for optimization."""
    def __init__(self,problemDimension,costfunc,numIter=1000,softConstraints=[np.inf,np.inf],randStartVals=[],phi1 = 4.05,phi2 = 4.05,chi = 0.7298):
        """Inputs:
            problemDimension: Number of parameters being iterated over. Future updates will read this directly.
            costfunc: Variable referencing the cost function. Can feed a function directly into this variable.
            numIter: In this version, optimization stops after a number of iterations. A future update will do costfunction change approaches.
            softConstraints: With evolutionary algorithms, soft constraints can be used. Position 0 is lower bound, position 1 upper bound. Set to np.inf if no constraints needed.
            randStartVals: Variable can be randomly or deterministically set as needed. This is the start condition for the algorithm. Recommend stochastic initialization for most cases.
            phi1,phi2 are search parameters for PSO. Suggest to keep at 4.05 unless playing with searching speeds. May lead to adverse behaviour if changed however.
            chi: The PSO dampening parameter. Suggest to keep at 0.7298 to prevent oscillatory behaviour.
            """
        super(WTAPSO, self).__init__()
        #Set instance variables here.
        self.problemDimension = problemDimension
        self.costfunc = costfunc
        self.numIter = numIter
        self.softConstraints = softConstraints
        self.randStartVals = randStartVals
        self.phi1 = phi1
        self.phi2 = phi2
        self.chi = chi
        self.swarmSize = 21                                          #For the moment, keep this at 21 for canonical WTAPSO. Updates will allow for arbitrarily large swarm sizes
        self.errorvec = np.ones(self.swarmsize)*np.inf               #Often helpful to keep track of fitness in an outside variable   
        self.minerr = 0                                              #This holds the min error value of the best (leader) agent.
        self.bestagent = []                                          #Holds the best agent, which is by definition, the swarm leader.
        self.nummem = 5                                              #The number of agents in a neighborhood swarm. Keep at 5 for the moment.
        self.neisize = np.floor(self.swarmsize/self.nummem)          #Defines the size of the neighborhoods here
        self.matrixdim = [self.nummem,4]                             #Size of matrices storing neighborhood info.     
        self.N = []                                                  #Contains information the location of each agent (except best agent) in the swarm. Rows are different neighborhoods
        self.errind = []                                             #Internal variable for holding locations of agents with given error values
        self.sizes = []
        self.neibest = np.inf                                        #Internal variable for holding a given neighborhood leader
        self.agenterrorbest = np.inf                                 #Internal variable for control of swarm updates
        self.agenterrbest = np.inf                                   #Internal variable for control of swarm updates
        self.row = 0                                                 #Internal Variable for control of swarm updates
        self.col = 0
    def createSwarm(self):
        """This function will create the swarm and assign control parameters to each agent"""
        self.swarm = [Particle(np.random.uniform(self.randStartVals[0],self.randStartVals[1],[self.problemDimension]),np.zeros(self.problemDimension),self.chi,self.phi1,self.phi2,np.inf) for ck in range(self.swarmsize)]   #Generate the swarm!!!
        for bc in range(self.swarmsize):                     #Set the agent phi and chi values
            self.swarm[bc].setChi(self.chi)
            self.swarm[bc].setPhi1(self.phi1)
            self.swarm[bc].setPhi2(self.phi2)

    def runSwarmInit(self):
        """This function is for the first run of the swarm. Sets fitness and position values"""
        for ii in range(self.swarmSize):
            curagent = self.swarm[ii]                        #Grab agent in the swarm
            curinputvals = curagent.returnPosition()      #Grab the inputs to the cost function
            curfitness = self.costfunc(curinputvals)         #Calculate a new fitness
            if curfitness < self.errorvec[ii]:
                self.swarm[ii].setFitness(curfitness)            #Set fitness for each agent
                self.swarm[ii].setBestPosition(curagent.returnPosition())       #At the first run, best position is the current position
                self.errorvec[ii] = swarm[ii].returnFitness()
    
    def createSwarmNet(self):
        """Function to create the swarm network and generate first best errors"""
        self.minerr = np.argmin(self.errorvec)                    #Grab the position of the best performing agent.
        self.bestagent = self.swarm[minerr] 
        self.genSwarmNetwork()     #Generate swarm matrix
        self.row = self.sizes[0]
        self.col = self.sizes[1]
        self.neibest = np.zeros(int(self.neisize))                     #Vector holding the agent leader values.
        self.updateNeighborErrors()

    def genSwarmNetwork(self):
        k = 0                             #Variable k is used in modulo to create network
        self.N = np.zeros([self.matrixdim[0],self.matrixdim[1]])        #N Holds our network
        for jj in range(self.swarmsize):           #Iterate through the agents and place them in the network
            if k <= self.nummem-1:
                self.N[k,int(np.mod(jj,self.neisize))] = jj
            else:
                k = 0                         #Reset mod factor
                self.N[k,int(np.mod(jj,self.neisize))] = jj
            k = k + 1
        #If agent 21 isn't a leader, then all agents are placed except 21. We swap it in for the global leader
        self.errind = np.where(self.N==self.minerr) #Find where best agent is in the matrix
        if self.minerr != self.swarmsize-1:
            self.N[self.errind[0],self.errind[1]] = self.swarmsize
        self.sizes = self.N.shape()
        #return [N,sizes,errind]
    def updateNeighborErrors(self):
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
        for kk in range(self.col):                     #Iterate through neighborhoods, which are columns of N
            agentvals = self.N[:,kk]
            errv = np.zeros(len(agentvals))
            for ll in range(len(agentvals)):
                curagent = self.swarm[int(agentvals[ll]-1)]
                errv[ll] = curagent.returnFitness()
            minern = np.where(errv==np.min(errv))
            self.neibest[kk] = agentvals[minern[0][0]]
        #minerr = np.argmin(errorvec)
        self.agenterrorbest = self.swarm[self.minerr]
        self.agenterrbest = self.agenterrorbest.returnFitness
        # [self.agenterrorbest,self.agenterrbest,self.neibest]
    
    def optimize(self):
        for ck in range(self.numIter):
            self.swarm = updateSwarm()         #Run the swarm update to update each agent
            #swarm = swarm[0]
            for bc in range(self.swarmsize):
                curagent = self.swarm[bc]                        #Grab agent in the swarm
                curinputvals = curagent.returnPosition()      #Grab the inputs to the cost function
                curfitness = self.costfunc(curinputvals)         #Calculate a new fitness
                if curfitness < self.errorvec[bc]:
                    self.swarm[bc].setFitness(curfitness)            #Set fitness for each agent
                    self.swarm[bc].setBestPosition(curagent.returnPosition())       #At the first run, best position is the current position
                    self.errorvec[bc] = self.swarm[bc].returnFitness()
                if bc == self.minerr:
                    print(self.errorvec[self.minerr])
            self.updateSwarmNetwork()
        #return [self.bestagent,self.errorvec,self.swarm]
    def updateSwarm(self):
        """ This function updates the swarm. Loops through and finds global and local leaders
        and reorganizes the swarm based on these values."""
        swarmup = self.swarm                     #Create an identical copy for update.
        neibesterrors = self.errorvec[self.neibest.astype(int)-1]       #Grab the neighborhood leader values
        neibesterr = np.min(neibesterrors)
        if np.size(self.neibesterr)>1:
            neibesterr = neibesterr[1]           #Check for duplicates
        errneibestindex = np.where(self.errorvec==neibesterr)
        bestneileader = self.swarm[errneibestindex[0][0]]
        self.bestagent.updateAgent(self.bestagent.returnBestPosition(),bestneileader.returnBestPosition(),self.constraints)  #Class function to update. See update description for how best, neibest, and regular agents are updated
        swarmup[self.minerr] = self.bestagent
        """Now update neighborhood leaders"""
        for nn in range(self.col):
            updateneibest = self.neibest[nn]
            update = swarmup[int(updateneibest)-1]         #Grab the neighborhood leader for each subswarm
            neighbor = self.N[:,nn]
            neierrvec = self.errorvec[neighbor.astype(int)-1]        #Vector containing errors of local leader agents
            neighbesterr = np.min(neierrvec)      #Now grab the error of the best neighborhood agent, use to grab best
            if np.size(neighbesterr)>1:
                neighbesterr = neibesterr[0]      #If multiple have same error value, just choose 1
            bestneigh = np.where(self.errorvec==neighbesterr)           #Find the location of the best neighbor in the swam
            updateneigh = self.swarm[bestneigh[0][0]-1]       #Now that we have found the best neighbor, we grab it from the swarm
            update.updateAgent(bestagent.returnBestPosition(),updateneigh.returnBestPosition(),self.constraints)
            swarmup[int(updateneibest)-1]=update
        """Now update the Neighborhood values"""
        for mm in range(self.col):               #We again poll the neighborhood
            currentneigh = self.N[:,mm]
            currentneiwhere = np.where(currentneigh!=neibest[mm])             #Search through the local neighborhood, ignoring the leader, we've already updated them
            currentneigh = currentneigh[currentneiwhere]            
            upneibest = self.swarm[int(neibest[mm])-1]       ##Since this agent is defined as a neighborhood leader, it will default be a part of the individual neighbor update
            for ii in range(len(currentneigh)):           #Update individual neighbors
                upvar = self.swarm[int(currentneigh[ii])-1]        #Grab the agent from the swarm
                neierrvecindwhere = np.where(currentneigh!=currentneigh[ii])
                neierrvecind = currentneigh[neierrvecindwhere]
                neierrvec = self.errorvec[(neierrvecind.astype(int))-1]        #Grab agent who is performing best
                neierrvec = np.min(neierrvec)
                if np.size(neierrvec) > 1:
                    neierrvec = neierrvec[0]              #If multiple occur, just choose one.
                bnei = np.where(errorvec==neierrvec)
                upN = self.swarm[bnei[0][0]]            #Pull agent from swarm
                upvar.updateAgent(upneibest.returnBestPosition(),upN.returnBestPosition(),constraints)
                swarmup[int(currentneigh[ii])-1] = upvar
        #return [swarmup,minerr]
        self.swarm = swarmup
def WTAPSO(problemDimension,numIter,costfunc,softConstraints,randStartVals,phi1 = 4.05,phi2 = 4.05,chi = 0.7298):
    """ Input Variables:
    problemDimension: Int value for dimension of the problem under optimization
    numIter: Int Maximum Number of iterations to run swarm for.
    costfunc: Variable pointing to a cost function for optimization
    softConstraints: 2D List or array containing softconstraints for optimization. softConstraints[0] = lower constraint, softConstraints[1] = upper constraints. Set to +- inf if not used.
    phi1,phi2: Float values partially determining growth rates.
    chi: Float value that works in concert with phi1,phi2 to control growth.
    randStartVals: Vector of low/high values between which to guess a random value to start swarm
    Output Variables:
    costFunctReturn: Output value of solved minimization or maximization.
    bestPosition: Output value of where on the solution space minimization or maximization occured.
    """
    """ Define all necessary includes for WTAPSO """
    """ WTAPSO constant definitions"""
    problemDim = problemDimension            #Dimension of the problem you'd like to solve
    numiterations = numIter                  #Number of iterations to search through swarm
    lowconst = softConstraints[0]            #Lower bound soft constraint for PSO
    highconst = softConstraints[1]           #Upper bound for soft constraint
    swarmsize = 21                           #For the moment, keep this fixed.
    #phi = phi1+phi2                          #Follow current convention on handeling phi1,2 values. Can modify to be more creative here.
    """Create the Swarm!"""
    swarm = [Particle(np.random.uniform(randStartVals[0],randStartVals[1],[problemDim]),np.zeros(problemDim),chi,phi1,phi2,np.inf) for ck in range(swarmsize)]   #Generate the swarm!!!
    for bc in range(swarmsize):                     #Set the agent phi and chi values
        swarm[bc].setChi(chi)
        swarm[bc].setPhi1(phi1)
        swarm[bc].setPhi2(phi2)
    errorvec = np.ones(swarmsize)*np.inf               #Often helpful to keep track of fitness in an outside variable
    """Do an initial run to get initial cost functional values"""
    for ii in range(swarmsize):
        curagent = swarm[ii]                        #Grab agent in the swarm
        curinputvals = curagent.returnPosition()      #Grab the inputs to the cost function
        curfitness = costfunc(curinputvals)         #Calculate a new fitness
        if curfitness < errorvec[ii]:
            swarm[ii].setFitness(curfitness)            #Set fitness for each agent
            swarm[ii].setBestPosition(curagent.returnPosition())       #At the first run, best position is the current position
            errorvec[ii] = swarm[ii].returnFitness()
    """ Create Swarm Network"""
    minerr = np.argmin(errorvec)                    #Grab the position of the best performing agent.
    bestagent = swarm[minerr]   
    nummem = 5                                      #Number of members in the local networks. Fix at 5 for the moment.
    neisize = np.floor(swarmsize/nummem)                #Defines the size of the neighborhoods here
    matrixdim = [nummem,4]                          #4 is fixed for the moment. Will update later
    [N,sizes,errind] = genSwarmNetwork(swarmsize,neisize,nummem,matrixdim,minerr)     #Generate swarm matrix
    row = sizes[0]                                  #Grab matrix dimensions for further processing
    col = sizes[1]
    neibest = np.zeros(int(neisize))                     #Vector holding the agent leader values.
    [agenterrorbest,agenterrbest,neibest] = updateNeighborErrors(swarm,N,neisize,neibest,minerr,swarmsize,errorvec,errind,row,col)
    """ Begin the Update Schema here"""
    for ck in range(numiterations):
        swarm = updateSwarm(swarm,errorvec,bestagent,neibest,row,col,N,minerr,softConstraints)         #Run the swarm update to update each agent
        swarm = swarm[0]
        for bc in range(swarmsize):
            curagent = swarm[bc]                        #Grab agent in the swarm
            curinputvals = curagent.returnPosition()      #Grab the inputs to the cost function
            curfitness = costfunc(curinputvals)         #Calculate a new fitness
            if curfitness < errorvec[bc]:
                swarm[bc].setFitness(curfitness)            #Set fitness for each agent
                swarm[bc].setBestPosition(curagent.returnPosition())       #At the first run, best position is the current position
                errorvec[bc] = swarm[bc].returnFitness()
            if bc == minerr:
                print(errorvec[minerr])
        [N,agenterrorbest,agenterrbest] = updateSwarmNetwork(swarm,swarmsize,neisize,nummem,matrixdim,minerr,N)
    return [bestagent,errorvec,swarm]


    