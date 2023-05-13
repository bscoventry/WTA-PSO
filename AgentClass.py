# -*- coding: utf-8 -*-
"""
Created on Wed Jul 04 10:54:31 2018

@author: Brandon S Coventry. Weldon School of Biomedical Engineering Purdue University
Purpose: This file contains all class definitions for WTAPSO
Revision History: 5/12/20 Added in GPU support for numpy calculations via cuPy
Notes: Reference: Coventry et al., 2017 J. Comput Neurosci
"""
import cupy as np              #Will use inf and other numpy functions. GPU support for numpy, change cupy to numpy for CPU support
import pdb
class Particle(object):
    """ A particle (also known as agent) within a swarm. Particles have the following
    properties.
    
    Attributes: 
        Fitness: Value of the fitness function for an individual agent. Defaults to infinity
        assuming global minimization. As all optimization problems can be reframed as 
        minimizations, this should be fine. Float datatype
        
        Position: Value which is used as an agent's current position in solution space. Used with velocity to update 
        agent trajectory
        
        Velocity: "Speed" at which agent is moving through the solution space. Used with position to update agent
        trajectory
        
        Phi1 and Phi2: Values used in updating agents positions in the solution space. Used to control swarm but
        also add stochasticity and jitter in agent positions
        
        Chi: Update value used to ensure motion of the the agent is "stable." Controls movement of agent.
        
    Class Functions:
        setFitness, setVelocity, setPosition, setVelocity, setPhi1, setPhi2, setChi. Used to set the respective values for
        each agent in the swarm.
        
        returnFitness, returnVelocity, returnPosition, returnVelocity, returnPhi1, returnPhi2, returnChi. Used to inspect
        the respective attributes current values
        
        updateAgent: Function which calculates new positions and velocities to move each agent through the solution space.
    """
    def __init__(self, position, velocity, chi, phi1, phi2, Fitness = np.inf):
        """ Return an agent with position *position* in the swarm 
        and initial Fitness of inf. Typical init function"""
        self.position = position
        self.bestPosition = position
        self.Fitness = Fitness
        self.chi = chi
        self.velocity = velocity
        self.phi1 = phi1
        self.phi2 = phi2
    def setFitness(self, newFitness):
        """ This class operation will update the fitness of the current agent. 
        Will be run after iterating through and recalculating fitnesses in the swarm.
        Note that update will only occur if new fitness is less than current fitness."""
        if newFitness <= self.Fitness:        
            self.Fitness = newFitness
        return self.Fitness
    
    def setPosition(self, newPosition):
        """ This class operation will update the agents current position in 
        for updating the PSO algorithm. Run after recalculation in the swarm"""
        self.position = newPosition
        return self.position

    def setBestPosition(self,newPosition):
        """ This class operation stores the agents best position for updating.
        Run during agent updates"""
        self.bestPosition = newPosition
        return self.bestPosition
    
    def setVelocity(self, newVelocity):
        """This class operation will update the agents current velocity in the
        PSO swarm update function. To be run after recalculating fitness"""
        self.velocity = newVelocity
        return self.velocity
    
    def setChi(self, newChi):
        """This class operation sets the chi value for swarm update. Usually
        chi is set, but use this if doing a swarm that can update its parameters
        each iteration. """
        self.chi = newChi
        return self.chi
    
    def setPhi1(self, newPhi1):
        """ This class operation sets the phi1 value for the swarm update. 
        This usually is set once, but can be used to actively update the parameters
        of the swarm each iteration. """
        self.phi1 = newPhi1
        return self.phi1
        
    def setPhi2(self, newPhi2):
        """ Same as setPhi1, but for Phi2"""
        self.phi2 = newPhi2
        return self.phi2
    
    def returnPosition(self):
        """ Class function to view agents current position value"""
        return self.position

    def returnBestPosition(self):
        """Class function to return the agents best position Value"""
        return self.bestPosition

    def returnVelocity(self):
        """Class function to view agents current velocity """
        return self.velocity
        
    def returnFitness(self):
        """Class function to view agents current fitness function value"""
        return self.Fitness
    
    def updateAgent(self,position1,position2,constraints):
        """Class function to update an agent. Leader 1 and Leader 2, and thus
        position1 and position2 will vary depending on which aspect of the swarm is being
        updated. 
        Global leaders: Leader 1 = global leader itself. Leader 2 = best performing local leader
        Local leader: Leader 1 = Global leader, Leader 2 = best performing from local swarm.
        Individual agents: Leader 1= Local neighborhood leader. Leader 2 = Best performing from local swarm.
        For clarificaiton, see Coventry et al., 2017 J. Comput Neurosci
        Pull positions from leaders for update
        To begin, generate random values from a uniform distribution between 0 and 1
        to modulate phi values"""
        randmod1 = np.random.rand(1)
        randmod2 = np.random.rand(1)
        uni1 = self.phi1*randmod1
        uni2 = self.phi2*randmod2
        newVelocity = self.chi*(self.velocity+uni1*(position1-self.position)+uni2*(position2-self.position))
        self.setVelocity(newVelocity)
        newPosition = newVelocity + self.position
        self.setPosition(newPosition)
        #Contraint handling
        constBoolHigh = np.where(np.asarray(newPosition)>np.asarray(constraints[1]))[0]         #Test for high constraints
        constBoolLow = np.where(np.asarray(newPosition)<np.asarray(constraints[0]))[0]
        #newPosition[constBoolHigh]=constraints[1][constBoolHigh]
        #newPosition[constBoolLow] = constraints[0][constBoolLow]
        #self.setPosition(newPosition)
        
        

