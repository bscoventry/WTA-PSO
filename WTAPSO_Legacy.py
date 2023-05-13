#Import the modules that are needed
#from neuron import *
#from nrn import *
#from SustainedFiring_PSO import *     #This is the sustained model
#from testloadspikes import *
import numpy as np
import random as rnd
#from InputPSTHs import *
import matplotlib.pyplot as plt
import pdb
#from ICModelAnalysis import *
import scipy.io
from testOptimizationFunctions import * #Import all for now. Can comment out if more memory is needed
# Constants and things like constants
#expdata = scipy.io.loadmat('14_Y12_track6_2670um_tuning1_54db_rates.mat')
#expdata = scipy.io.loadmat('13_A8_track4_770um_tuning2_70db_rates.mat')
#numE = 1                              #For PSO model, we lump Ex inputs
#numI = 1                              #For PSO model, we lump In inputs
#cfe = input('Excitatory Input CF: ')
#cfe = float(cfe)
#cfi = input('Inhibitory Input CF: ') #Now an optimization parameter
#cfi = float(cfi)
#latency = 5                           #Input PSTH latency
#iterationnum = 50                     #Sets the total number of iterations of the swarm algorithm
#yoo = input('Young(0) or Aged(1) model: ')    #Young or aged model
# OUnoise vector: [EX conductance, In Conductance, EX sd, In sd, Ex tau, In tau]
#if yoo == 0:
    #OUnoise = [0.000935,0.0005,0.0008891,0.0005,2,10]
#else:
    #OUnoise = [0.000935,0.00047,0.0008891,0.0005,2,10]
#Etype = 2                       #1 MSO, 2 DCN, 3 LSO 4 VCN
#Itype = 1                       # 1 DNLL 2 VNLL 
#networkcheck = 0                      #Network run 1, single 0
ssize = 21                            #Swarm size
nummem = 5                            #Size of local neighborhoods
numtrials = 5                         #Number of trial repititions
#sgi = np.array([500,525,574,615,659,707,757,812,870,933,1000,1071,1148,1231,1319,1414,1515,1624,1741,1866,2000,2143,2297,2462,2639,2828,3031,3249,3482,3732,4000,4287,4594,4924,5278,5656,6062,6498,6964,7464,8000,8574,9189,9849,10556,11313,12125,12996,13928,14928,16000,17148,18379,19698,21112,22627,24251,25992,27857,29857,32000,34296,36758,39396])
#sgi = sgi.astype(float)
#sgi = sgi[0:29]
# Initiate SWARM!
chi = .7298                          #Constriction coefficients. See Poli et al 2007
phi1 = 2.05 
phi2 = 2.05
phi = phi1+phi2
#GABAbound = 100                     #Soft bound on GABA inhibition
swarm = []                          #List of vectors for swarm
costfunc = lambda inputvec: Rosenbrock(inputvec)     #This is the function you wish to optimize. Takes an inputvector of values. 
inSize = 100
iterationnum = 1000
for ii in range(0,ssize):
    s = np.zeros([inSize,5],dtype=float)
    for jj in range(inSize):
        s[jj,0] = rnd.randint(-500,500)
        s[jj,1] = 0
    # s[0,0] = rnd.randint(120,170)  #Excitatory AMPA conductance initial input
    # s[1,0] = rnd.randint(120,170)  #Excitatory NMDA conductance intial input
    # s[2,0] = rnd.randint(120,170)  #Inhibitory GABA conductance initial input
    # s[3,0] = rnd.randint(-70,-50)  #Initial membrane potential
    # s[4,0] = rnd.randint(5,7)      #Ex Input Q10 factor
    # s[5,0] = rnd.randint(110,120)    #Ex Input firing rate factor
    # s[6,0] = rnd.randint(5,7)      #In input Q10 factor
    # s[7,0] = rnd.randint(70,90)    #In input firing rate
    # s[8,0] = cfe                   #Difference between excitation and inhibition (EX is fixed)
    # s[9,0] = rnd.randint(-2,2)     #Inhibitory lag
    # s[10,0] = rnd.randint(0,5)     #Ex A1
    # s[11,0] = rnd.randint(0,5)     #Ex B1
    # s[12,0] = rnd.randint(37,55)   #Ex A2
    # s[13,0] = rnd.randint(37,55)   #Ex B2
    # s[14,0] = rnd.randint(10,23)   #Ex A3
    # s[15,0] = rnd.randint(10,23)   #Ex B3
    # s[16,0] = rnd.randint(1,12)    #Ex A4
    # s[17,0] = rnd.randint(1,12)    #Ex B4
    # s[18,0] = rnd.randint(0,5)     #In A1
    # s[19,0] = rnd.randint(0,5)     #In B1
    # s[20,0] = rnd.randint(37,55)   #In A2
    # s[21,0] = rnd.randint(37,55)   #In B2
    # s[22,0] = rnd.randint(10,23)   #In A3
    # s[23,0] = rnd.randint(10,23)   #In B3
    # s[24,0] = rnd.randint(1,12)    #In A4
    # s[25,0] = rnd.randint(1,12)    #In B4
    # s[0,1] = rnd.randint(10,20)    #Excitatory AMPA condunctance initial velocity
    # s[1,1] = rnd.randint(10,20)    #Excitatory NMDA conductance initial velocity
    # s[2,1] = rnd.randint(10,20)    #Inhibitory GABA conductance initial velocity
    # s[3,1] = 0                     #Initial velocity for mem potential 
    # s[4,1] = 0                     #Ex Input Q10 velocity
    # s[5,1] = 0                     #Ex Input firing rate velocity
    # s[6,1] = 0                     #In input Q10 velocity
    # s[7,1] = 0                     #In input firing rate velocity
    # s[8,1] = 0                     #EX/IN difference velocity (EX is fixed)
    # s[9,1] = 0                     #Inhibitory lag
    # s[10,1] = 0                    #Ex A1
    # s[11,1] = 0                    #Ex B1
    # s[12,1] = 0                    #Ex A2
    # s[13,1] = 0                    #Ex B2
    # s[14,1] = 0                    #Ex A3
    # s[15,1] = 0                    #Ex B3
    # s[16,1] = 0                    #Ex A4
    # s[17,1] = 0                    #Ex B4
    # s[18,1] = 0                    #In A1
    # s[19,1] = 0                    #In B1
    # s[20,1] = 0                    #In A2
    # s[21,1] = 0                    #In B2
    # s[22,1] = 0                    #In A3
    # s[23,1] = 0                    #In B3
    # s[24,1] = 0                    #In A4
    # s[25,1] = 0                    #In B4
    s[0,2] = np.inf                   #Initial error is infinity
    swarm.append(s)    
errorvec = np.ones(ssize)*np.inf

#Load model parameters
#vdepscale = 35.7
#APPDTau = 25.42
#PPDRatio = 0.6629
#gTau = 15
#gTau1 = 3
#tprime = ((gTau*gTau1)/(gTau-gTau1))*np.log(gTau/gTau1)
#gscale = 1/(np.exp(-tprime/15)-np.exp(-tprime/3))
#GPPDTau = APPDTau * PPDRatio
#Input PSTH Generation Goes here

#More model parameters
# AMPAtau1 = .5464
# AMPAtau2 = 6
# NMDAtau1 = 32
# NMDAtau2 = 50
# GABAtau1 = 3
# GABAtau2 = 15
# GABAB = 0           #Not relevant at this point
# inAMPA1 = 5         #These will be chosen based on model fits
# inAMPA2 = 6
# inNMDA1 = 3
# inNMDA2 = 1.5
# inGABA1 = 3
# inGABA2 = 4
#BiasCur = 0.006564*vinit + .459

#Initial Run: Set up the Swarm
#Loop to iterate through all agents for every trial for the swarm
for pp in range(ssize):          #Loop through each agent
    agent = swarm[pp]              # Grab current agent
    # perAMPA = agent[0,0]           #Grab current AMPA conductance
    # perNMDA = agent[1,0]           #Grab current NMDA conductance
    # perGABA = agent[2,0]           #Grab current GABA conductance
    # AMPA1 = inAMPA1*(perAMPA/100.)
    # NMDA1 = inNMDA1*(perNMDA/100.)
    # GABA1 = inGABA1*(perGABA/100.)
    # E = [AMPA1,NMDA1,GABA1,GABAB]   #Not sure I need this
    # Biascur = 0.006564*agent[3,0]   #Generate Bias current
    # cfi = agent[8,0]                #Set inhibitory cf. Cotuned at first
    # ExQ10 = agent[4,0]              #Set input characteristics
    # Exrate = agent[5,0]
    # InQ10 = agent[6,0]
    # Inrate = agent[7,0]
    # if Etype == 1:
    #     #[Exrate Exmaxrate]
    #     AMPA = 5
    #     NMDA = 3
    # elif Etype == 2:
    #     #[Exrate Exmaxrate]
    #     AMPA =6
    #     NMDA = 1.5
    # elif Etype == 3:
    #     #[Exrate Exmaxrate]
    #     AMPA = 5
    #     NMDA = 1.5
    # elif Etype == 4:
    #     #[Exrate Exmaxrate]
    #     AMPA = 5
    #     NMDA = 1.5
    # maxrateE = Exrate
    # maxrateI = Inrate
    # fsl = np.zeros([numtrials,len(sgi)])
    # PerTrialSpk = np.zeros([numtrials,len(sgi)])
    # onset = 200.
    # stimduration = 750.
    # spikevec = [0 for i in range(len(sgi))]
    # for nn in range(len(sgi)):
    #     #Generate Excitatory characteristics first
    #     rateE = FTCcharsfreq(sgi[nn],cfe,sgi,agent[4,0],agent[5,0])    
    #     rateE = rateE[0]              #Do not need FTC data at this point
    #     if np.isnan(rateE):
    #         rateE = 0
    #     #pdb.set_trace()
    #     psthe,binvece = PSTHgen(latency,rateE,agent[5,0],agent[10,0],agent[11,0],agent[12,0],agent[13,0],agent[14,0],agent[15,0],agent[16,0],agent[17,0])
    #     #Now Inhibitory
    #     rateI = FTCcharsfreq(sgi[nn],agent[8,0],sgi,agent[6,0],agent[7,0])
    #     rateI = rateI[0]
    #     if np.isnan(rateI):
    #         rateI = 0
    #     psthi,binveci = PSTHgen(latency,rateI,agent[7,0],agent[18,0],agent[19,0],agent[20,0],agent[21,0],agent[22,0],agent[23,0],agent[24,0],agent[25,0])
    #     #cdfe = makeCDF(psthe)
    #     trialvec = [0 for j in range(numtrials)]
    #     for mm in range(numtrials):
    #         cdfe = makeCDF(psthe)
    #         drvinputs = InvTransform(sgi[nn],cdfe,rateE)
    #         cdfi = makeCDF(psthi)
    #         drvIinputs = InvTransform(sgi[nn],cdfi,rateI)
    #         randseed = rnd.uniform(0, 1)          #Generate random seed for OU noise process
    #         randseed = round(randseed*100)        #Get it into a form OU likes
    #         [spkts, voltages] = SustainedFiring_PSO(numE,numI,OUnoise,networkcheck,randseed,drvinputs,drvIinputs,APPDTau,GPPDTau,gTau,gscale,vdepscale,agent[3,0],Biascur,E,AMPAtau1,AMPAtau2,NMDAtau1,NMDAtau2,GABAtau1,GABAtau2)
    #         [spike_timesRAW, num_spikes] = SpDetect(voltages[0])             #Count and detect spikes
    #         #Latency Calculations
    #         for zz in range(num_spikes):
    #             if spike_timesRAW[zz] >= 200.:
    #                 fsl[mm,nn] = spike_timesRAW[zz] - onset
    #                 break            #Once first spike is found, no need to continue in loop
    #         spike_timesRa = np.array([],dtype = 'float')
    #         offset = np.array([],dtype = 'float')
    #         for uu in range(num_spikes):
    #             if (spike_timesRAW[uu] > onset + stimduration):
    #                 offset = np.append(offset,spike_timesRAW[uu])
    #         offset = offset - stimduration
    #         spike_timesRa = np.array([],dtype = 'float')
    #         for yy in range(num_spikes):
    #             #pdb.set_trace()
    #             if spike_timesRAW[yy] > 0: 
    #                 if spike_timesRAW[yy] <= stimduration+onset:
    #                     spike_timesRa = np.append(spike_timesRa,spike_timesRAW[yy])
    #         if len(spike_timesRa) > 0:
    #             spike_times = spike_timesRa - onset
    #         else:
    #             spike_times = []
    #         trialvec[mm] = spike_timesRa
    #     spikevec[nn] = trialvec
    #     #pdb.set_trace()
    # [meanvec,sdvec] = ICstats(spikevec,numtrials,sgi)
    # sevec = sdvec/np.sqrt(numtrials)            #Set up standard error
    # FTCgen(meanvec,sevec,sgi)
    # #pdb.set_trace()
    Fitfun = costfunc(agent[:,0])
    #FTC generation
    #Hold best positions in column 5 of swarm vector
    for rr in range(inSize):
        agent[rr,4] = agent[rr,0]
    if Fitfun < errorvec[pp]:
        errorvec[pp] = Fitfun
        agent[0,3] = Fitfun
        agent[0,2] = Fitfun
    swarm[pp] = agent
#Define Neighborhoods
#N is a matrix which contains placement of each swarm member. I.E. columns correspond to 4 neighborhoods. Down the column, agents are placed into select neighborhoods. This is a way to map agent locations in the neighborhoods.
minerr = np.argmin(errorvec)
bestagent = swarm[minerr]
neisize = int(np.floor(ssize/nummem))
k = 0               #Starting point for neighborhood generation
N = np.zeros([5,4])
for jj in range(ssize):
    if k <= nummem-1:              #Use this as a "wrap aroudn factor"
        N[k,np.mod(jj,neisize)] = jj          #Modulo function takes care of spacing
    else:
        k = 0
        N[k,np.mod(jj,neisize)] = jj          #Reset mod factor
    k = k + 1
#If agent 21 isn't a leader, Then all agents are placed except 21. We need to swap him in for the global leader.
errind = np.where(N == minerr)             #Find where the best agent is in the matrix
N[errind[0],errind[1]] = ssize             # Replace the best (it will have its own holder) with agent 21
sizes = N.shape
row = sizes[0]                              #Get the matrix dimensions
col = sizes[1]
neibest = np.zeros(neisize)
for kk in range(col):                     #Iterate through neighborhoods, which are columns of N
    curcol = kk
    agentvals = N[:,kk]
    errv = np.zeros(len(agentvals))
    for ll in range(len(agentvals)):
        curagent = swarm[int(agentvals[ll]-1)]
        errv[ll] = curagent[0,2]
    minern = np.where(errv==np.min(errv))
    neibest[kk] = agentvals[minern[0][0]]
agenterrorbest = swarm[minerr]
agenterrbest = agenterrorbest[0,3]
#Begin update scheme
#Update head hauncho first
ploterr = []
for yy in range(iterationnum):
    swarmup = swarm                       #Always a good idea to recreate an indentical copy
    bestagent = swarmup[minerr]           #Grab the current swarm member
    neibesterrors = errorvec[neibest.astype(int)-1]     #Grab neighborhood leader values
    neibesterr = np.min(neibesterrors)
    if np.size(neibesterr)>1:
        neibesterr = neibesterr[1]            #In case of duplicate, choose first one
    errneibestindex = np.where(errorvec==neibesterr)
    bestneileader = swarm[errneibestindex[0][0]]
    uni = np.zeros(2)
    uni[0] = phi1*np.random.rand(1)
    uni[1] = phi2*np.random.rand(1)
    newvi = chi*(bestagent[:,1]+uni[0]*(bestagent[:,4]-bestagent[:,0])+uni[1]*(bestneileader[:,4]-bestagent[:,0]))
    newxi = newvi+bestagent[:,0]
    #Make sure that xi is not zero at any point
    # if newxi[0] < 0:               #If AMPA,NMDA, or GABA conductances are less than zero, set them to zero
    #     newxi[0] = 0
    # if newxi[1] < 0:
    #     newxi[1] = 0
    # if newxi[2] < 0:
    #     newxi[1] = 0
    # if newxi[0] > 150:
    #     newxi[0] = 150
    # if newxi[1] > 150:
    #     newxi[1] = 150
    # if newxi[2] > 150:
    #     newxi[2] = 150
    # if newxi[3] < -80:
    #     newxi[3] = -80
    # if newxi[3] > -53:
    #     newxi[3] = -53
    bestagent[:,0] = newxi        #Update agent parameters
    bestagent[:,1] = newvi
    swarmup[minerr] = bestagent        #Now store this agent in the new swarm error placement
    #Next update neighborhood leaders
    for nn in range(col):           #Iterate over columns of N update matrix
        updateneibest = neibest[nn]
        update = swarm[int(updateneibest)-1]
        neighbor = N[:,nn]          #Grab neighborhood agents
        neierrvec = errorvec[neighbor.astype(int)-1]
        neighbesterr = np.min(neierrvec)
        if np.size(neighbesterr)>1:
            neighbesterr = neighbesterr[0]         #If multiple choose first one
        bestneigh = np.where(errorvec==neighbesterr)
        updateneigh = swarm[bestneigh[0][0]-1] #Now that we have found the best neighbor, we grab him from the swarm
        uni = np.zeros(2)
        uni[0] = phi1*np.random.rand(1)
        uni[1] = phi2*np.random.rand(1)
        newvi = chi*(update[:,1]+uni[0]*(bestagent[:,4]-update[:,0])+uni[1]*(updateneigh[:,4]-update[:,0]))
        newxi = newvi + update[:,0]           #Update position
        # if newxi[0] < 0:
        #     newxi[0] = 0
        # if newxi[0] > 150:
        #     newxi[0] = 150
        # if newxi[1] > 150:
        #     newxi[1] = 150
        # if newxi[2] > 150:
        #     newxi[2] = 150
        # if newxi[1] < 0:
        #     newxi[1] = 0
        # if newxi[2] < 0:
        #     newxi[2] = 0
        # if newxi[3] < -80:
        #     newxi[3] = -80
        # if newxi[3] > -53:
        #     newxi[3] = -53
        update[:,0] = newxi
        update[:,1] = newvi
        swarmup[int(updateneibest)-1] = update
    #Finally, update Neighborhood values
    for mm in range(col):           #We are going to again poll the neighborhood
        currentneigh = N[:,mm]
        currentneiwhere = np.where(currentneigh!=neibest[mm])
        currentneigh = currentneigh[currentneiwhere]
        upneibest = swarm[int(neibest[mm])-1]          #Since this guy is defined as a neighborhood leader, he will default be a part of the individual neighbor update
        for ii in range(len(currentneigh)):     #Update individual neighbors
            upvar = swarm[int(currentneigh[ii])-1]     #Grab the agent from the swarm
            neierrvecindwhere = np.where(currentneigh!=currentneigh[ii])
            neierrvecind = currentneigh[neierrvecindwhere]
            neierrvec = errorvec[(neierrvecind.astype(int))-1]
            neierrvec = np.min(neierrvec)       #Grab agent who is performing best
            if np.size(neierrvec) > 1:
                neierrvec = neierrvec[0]       #If multiple, grab the first one
            bnei = np.where(errorvec==neierrvec)
            upN = swarm[bnei[0][0]]                   #Pull agent from swarm
            uni = np.zeros(2)
            uni[0] = phi1*np.random.rand(1)
            uni[1] = phi2*np.random.rand(1)
            newvi = chi*(upvar[:,1]+uni[0]*(upneibest[:,4]-upvar[:,0])+uni[1]*(upN[:,4]-upvar[:,0]))
            newxi = newvi+upvar[:,0]
            # if newxi[0]<0:
            #     newxi[0] = 0
            # if newxi[1] < 0:
            #     newxi[1] = 0
            # if newxi[2] < 0:
            #     newxi[2] = 0
            # if newxi[0] > 150:
            #     newxi[0] = 150
            # if newxi[1] > 150:
            #     newxi[1] = 150
            # if newxi[2] > 150:
            #     newxi[2] = 150
            # if newxi[3] < -80:
            #     newxi[3] = -80
            # if newxi[3] > -53:
            #     newxi[3] = -53
            upvar[:,0] = newxi
            upvar[:,1] = newvi
            swarmup[int(currentneigh[ii])-1] = upvar
    swarm = swarmup
    for pp in range(ssize):
        agent = swarm[pp]              # Grab current agent
        # perAMPA = agent[0,0]           #Grab current AMPA conductance
        # perNMDA = agent[1,0]           #Grab current NMDA conductance
        # perGABA = agent[2,0]           #Grab current GABA conductance
        # AMPA1 = inAMPA1*(perAMPA/100.)
        # NMDA1 = inNMDA1*(perNMDA/100.)
        # GABA1 = inGABA1*(perGABA/100.)
        # E = [AMPA1,NMDA1,GABA1,GABAB]   #Not sure I need this
        # Biascur = 0.006564*agent[3,0]   #Generate Bias current
        # cfi = agent[8,0]                #Set inhibitory cf. Cotuned at first
        # ExQ10 = agent[4,0]              #Set input characteristics
        # Exrate = agent[5,0]
        # InQ10 = agent[6,0]
        # Inrate = agent[7,0]
        # if Etype == 1:
        #     #[Exrate Exmaxrate]
        #     AMPA = 5
        #     NMDA = 3
        # elif Etype == 2:
        #     #[Exrate Exmaxrate]
        #     AMPA =6
        #     NMDA = 1.5
        # elif Etype == 3:
        #     #[Exrate Exmaxrate]
        #     AMPA = 5
        #     NMDA = 1.5
        # elif Etype == 4:
        #     #[Exrate Exmaxrate]
        #     AMPA = 5
        #     NMDA = 1.5
        # maxrateE = Exrate
        # maxrateI = Inrate
        # fsl = np.zeros([numtrials,len(sgi)])
        # PerTrialSpk = np.zeros([numtrials,len(sgi)])
        # onset = 200.
        # stimduration = 750.
        # spikevec = [0 for i in range(len(sgi))]
        # for nn in range(len(sgi)):
        #     #Generate Excitatory characteristics first
        #     rateE = FTCcharsfreq(sgi[nn],cfe,sgi,agent[4,0],agent[5,0])    
        #     rateE = rateE[0]              #Do not need FTC data at this point
        #     if np.isnan(rateE):
        #         rateE = 0
        #     #pdb.set_trace()
        #     psthe,binvece = PSTHgen(latency,rateE,agent[5,0],np.maximum(0.00001,agent[10,0]),np.maximum(0.00001,agent[11,0]),np.maximum(0.00001,agent[12,0]),np.maximum(0.00001,agent[13,0]),np.maximum(0.00001,agent[14,0]),np.maximum(0.00001,agent[15,0]),np.maximum(0.00001,agent[16,0]),np.maximum(0.00001,agent[17,0]))
        #     #Now Inhibitory
        #     rateI = FTCcharsfreq(sgi[nn],agent[8,0],sgi,agent[6,0],agent[7,0])
        #     rateI = rateI[0]
        #     if np.isnan(rateI):
        #         rateI = 0
        #     psthi,binveci = PSTHgen(latency,rateI,agent[7,0],np.maximum(0.00001,agent[18,0]),np.maximum(0.00001,agent[19,0]),np.maximum(0.00001,agent[20,0]),np.maximum(0.00001,agent[21,0]),np.maximum(0.00001,agent[22,0]),np.maximum(0.00001,agent[23,0]),np.maximum(0.00001,agent[24,0]),np.maximum(0.00001,agent[25,0]))
        #     #cdfe = makeCDF(psthe)
        #     trialvec = [0 for j in range(numtrials)]
        #     for mm in range(numtrials):
        #         cdfe = makeCDF(psthe)
        #         drvinputs = InvTransform(sgi[nn],cdfe,rateE)
        #         cdfi = makeCDF(psthi)
        #         drvIinputs = InvTransform(sgi[nn],cdfi,rateI)
        #         randseed = rnd.uniform(0, 1)          #Generate random seed for OU noise process
        #         randseed = round(randseed*100)        #Get it into a form OU likes
        #         [spkts, voltages] = SustainedFiring_PSO(numE,numI,OUnoise,networkcheck,randseed,drvinputs,drvIinputs,APPDTau,GPPDTau,gTau,gscale,vdepscale,agent[3,0],Biascur,E,AMPAtau1,AMPAtau2,NMDAtau1,NMDAtau2,GABAtau1,GABAtau2)
        #         [spike_timesRAW, num_spikes] = SpDetect(voltages[0])             #Count and detect spikes
        #         #Latency Calculations
        #         for zz in range(num_spikes):
        #             if spike_timesRAW[zz] >= 200.:
        #                 fsl[mm,nn] = spike_timesRAW[zz] - onset
        #                 break            #Once first spike is found, no need to continue in loop
        #             spike_timesRa = np.array([],dtype = 'float')
        #             offset = np.array([],dtype = 'float')
        #         for uu in range(num_spikes):
        #             if (spike_timesRAW[uu] > onset + stimduration):
        #                 offset = np.append(offset,spike_timesRAW[uu])
        #             offset = offset - stimduration
        #             spike_timesRa = np.array([],dtype = 'float')
        #         for qq in range(num_spikes):
        #             #pdb.set_trace()
        #             if spike_timesRAW[qq] > 0: 
        #                 if spike_timesRAW[qq] <= stimduration+onset:
        #                     spike_timesRa = np.append(spike_timesRa,spike_timesRAW[qq])
        #             if len(spike_timesRa) > 0:
        #                 spike_times = spike_timesRa - onset
        #             else:
        #                 spike_times = []
        #         trialvec[mm] = spike_timesRa
        #     spikevec[nn] = trialvec
        # [meanvec,sdvec] = ICstats(spikevec,numtrials,sgi)
        # sevec = sdvec/np.sqrt(numtrials)            #Set up standard error
        # FTCgen(meanvec,sevec,sgi)
        #         #pdb.set_trace()
        Fitfun = costfunc(agent[:,0])
        # if Fitfun < np.min(errorvec):
        #     bestpsthe = psthe
        #     bestpsthi = psthi
        #     bestrate = meanvec
        #     bestsd = sdvec
        if Fitfun < errorvec[pp]:
            errorvec[pp] = Fitfun
            for rr in range(inSize):
                agent[rr,4] = agent[rr,0]
            agent[0,2] = Fitfun
        swarm[pp] = agent
        k = 0               #Starting point for neighborhood generation
        for jj in range(ssize):
            if k <= nummem-1:              #Use this as a "wrap aroudn factor"
                N[k,np.mod(jj,neisize)] = jj          #Modulo function takes care of spacing
            else:
                k = 0
                N[k,np.mod(jj,neisize)] = jj          #Reset mod factor
            k = k + 1
                #If agent 21 isn't a leader, Then all agents are placed except 21. We need to swap him in for the global leader.
        errind = np.where(N == minerr)             #Find where the best agent is in the matrix
        N[errind[0],errind[1]] = ssize             # Replace the best (it will have its own holder) with agent 21
        sizes = N.shape
        row = sizes[0]                              #Get the matrix dimensions
        col = sizes[1]
        neibest = np.zeros(neisize)
        for kk in range(col):                     #Iterate through neighborhoods, which are columns of N
            curcol = kk
            agentvals = N[:,kk]
            errv = np.zeros(len(agentvals))
            for ll in range(len(agentvals)):
                curagent = swarm[int(agentvals[ll])-1]
                errv[ll] = curagent[0,2]
            minern = np.where(errv==np.min(errv))
            neibest[kk] = agentvals[minern[0][0]]
        minerr = np.argmin(errorvec)
        agenterrorbest = swarm[minerr]
        agenterrbest = agenterrorbest[0,2]
print(min(errorvec))
# figfinal = plt.show()
# bx = plt.gca()
# bx.errorbar(sgi,bestrate,yerr=bestsd)
# bx.errorbar(sgi,expdata["sgirate"][0][0:29],yerr=expdata["sgistd"][0][0:29])
# bx.set_xscale('log')
# plt.legend(['Model', 'Experimental'], loc='upper right')
# plt.show()
# inputpe = plt.show()
# cx = plt.gca()
# cx.plot(bestpsthe)
# inputpi = plt.show()
# dx = plt.gca()
# dx.plot(bestpsthi)