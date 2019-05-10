def sdeWrapper( modelParams, expParams, featureArray ):
    '''
    Prepares for and runs the SDE time-stepped evolution of neural firing rates.
    Inputs:
        1. modelParams: struct with connection matrices etc
        2. expParams: struct with timing info about experiment, eg when stimuli are given.
        3. featureArray: numFeatures x numStimsPerClass x numClasses array of stimuli
    Output:
        1. simResults: EN timecourses and final P2K and K2E connection matrices.
          Note that other neurons' timecourses (outputted from sdeEvolutionMnist)
          are not retained in simResults.

    #---------------------------------------------------------------------------

    4 sections:
        1. load various params needed for pre-evolution prep
        2. specify stim and octo courses
        3. interaction equations and step through simulation
        4. unpack evolution output and export
    '''

    import numpy as np
    from support_functions.sdeEvo import sdeEvoMNIST

    ## 1. initialize states of various components:

    # unpack a few variables that are needed before the evolution stage:
    nP = modelParams.nP # = nG
    nG = modelParams.nG
    nPI = modelParams.nPI
    nK = modelParams.nK
    nR = modelParams.nR # = nG
    nE = modelParams.nE
    F2R = modelParams.F2R

    ##  2b. Define Stimuli and Octopamine time courses:

    # set time span and events:
    simStart = expParams.simStart
    simStop =  expParams.simStop
    timeStep = 2*0.01

    total_steps = (simStop - timeStep - simStart)/timeStep
    time = np.around(np.linspace(simStart, simStop, total_steps + 1), decimals=4)

    # create stimMags, a matrix n x m where n = # of odors and m = # timesteps.
    stimStarts = expParams.stimStarts
    durations = expParams.durations
    whichClass = expParams.whichClass

    classList = np.sort(np.unique(whichClass))
    classMags = expParams.classMags
    # create a classMagMatrix, each row giving the stimulus magnitudes of a different class:
    classMagMatrix = np.zeros((len(classList), len(time))) # ie 4 x len(time)
    for i,cl in enumerate(classList):
        # extract the relevant odor puffs. All vectors should be same size, in same order
        puffs = (whichClass == cl)
        theseClassStarts = stimStarts[puffs]
        theseDurations = durations[puffs]
        theseMags = classMags[puffs]

        for j in range(len(theseClassStarts)):
            cols = (theseClassStarts[j] < time) & (time < (theseClassStarts[j] + theseDurations[j]))
            classMagMatrix[i, cols] = theseMags[j]

    print(f'FOLLOW-UP[{__file__}]')
    print(len(np.nonzero(cols)[0])) # Contains one element too many
    # NEED TO FIX?: Doesn't correspond to matlab counterpart
    # python version: [71250:71260] length=11
    # matlab version: [71252:71261] length=10
    # import pdb; pdb.set_trace()

    # Apply a lowpass to round off the sharp start-stop edges of stimuli and octopamine:
    lpParam = expParams.lpParam # default transition zone = 0.12 sec
    L = round(lpParam/timeStep) # define the slope of low pass transitions here
    lpWindow = np.hamming(L) # window of width L
    lpWindow /= lpWindow.sum()

    # window the stimulus time courses:
    for i in range(len(classList)):
        classMagMatrix[i,:] = np.convolve(classMagMatrix[i,:], lpWindow, 'same')

    # window the octopamine:
    octoMag = expParams.octoMag
    octoHits = np.zeros(len(time))
    # octoStart = expParams.octoStart
    # durationOcto = expParams.durationOcto
    octoStop = [i + expParams.durationOcto for i in expParams.octoStart]
    for i in range(len(expParams.octoStart)):
        hits = (time >= expParams.octoStart[i]) & (time < octoStop[i])
        octoHits[ hits ] = octoMag
    octoHits = np.convolve(octoHits, lpWindow, 'same') # the low pass filter

    ## do SDE time-step evolution:

    # Use euler-maruyama SDE method, milstein's version.
    #  Y (the vector of all neural firing rates) is structured as a row vector as follows: [ P, PI, L, K, E ]
    Po = np.ones(nP) # P are the normalized FRs of the excitatory PNs
    PIo = np.ones(nPI) # PI are the normed FRs of the inhib PNs
    Lo = np.ones(nG)
    Ro = modelParams.Rspont
    Ko = np.ones(modelParams.nK) # K are the normalized firing rates of the Kenyon cells
    Eo = np.zeros(modelParams.nE) # start at zeros
    initCond = np.concatenate((Po, PIo, Lo, Ro, Ko, Eo) , axis=None) # initial conditions for Y

    tspan = [ simStart, simStop ]
    seedValue = 0 # to free up or fix randn
    # If = 0, a random seed value will be chosen. If > 0, the seed will be defined.

    # Run the SDE evolution:
    thisRun = sdeEvoMNIST(tspan, initCond, time, classMagMatrix, featureArray,
        octoHits, modelParams, expParams, seedValue )
    # Time stepping is now done.

    ## Unpack Y and save results:
    # Y is a matrix numTimePoints x nG. Each col is a PN, each row holds values for a single timestep
    Y = thisRun['Y']

    # save some inputs and outputs to a struct for argout:
    simResults = dict()
    simResults['T']= thisRun['T'] # timing information
    simResults['E'] =  thisRun['E']
    simResults['octoHits'] = octoHits
    simResults['K2Efinal'] = thisRun['K2Efinal']
    simResults['P2Kfinal'] = thisRun['P2Kfinal']

    if modelParams.saveAllNeuralTimecourses:
        simResults['P'] = Y[:,:nP]
        simResults['K'] = Y[:, nP + nPI + nG + nR + 1: nP + nPI + nG + nR + nK]

        # other neural timecourses
        # simResults['PI'] = Y[:,nP + 1:nP + nPI]
        # simResults['L'] = Y[:, nP + nPI + 1:nP + nPI + nG]
        # simResults['R'] = Y[:, nP + nPI + nG + 1: nP + nPI + nG + nR]

    return simResults
