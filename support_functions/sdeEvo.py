def sdeWrapper( modelParams, expParams, featureArray ):
    # Prepares for and runs the SDE time-stepped evolution of neural firing rates.
    # Inputs:
    #   1. modelParams: struct with connection matrices etc
    #   2. expParams: struct with timing info about experiment, eg when stimuli are given.
    #   3. featureArray: numFeatures x numStimsPerClass x numClasses array of stimuli
    # Output:
    #   1. simResults: EN timecourses and final P2K and K2E connection matrices.
    #       Note that other neurons' timecourses (outputted from sdeEvolutionMnist)
    #       are not retained in simResults.

#-------------------------------------------------------------------------------

    # 4 sections:
    # 1. load various params needed for pre-evolution prep
    # 2. specify stim and octo courses
    # 3. interaction equations and step through simulation
    # 4. unpack evolution output and export

    import numpy as np

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
    for i,c in enumerate(classList):
        # extract the relevant odor puffs. All vectors should be same size, in same order

        puffs = (whichClass == c)
        theseClassStarts = stimStarts[puffs]
        theseDurations = durations[puffs]
        theseMags = classMags[puffs]
        for j in range(len(theseClassStarts)):

            # print('foo')
            # print('theseClassStarts[j]:', theseClassStarts[j])
            # print((theseClassStarts[j] + theseDurations[j]))
            # print('time type:',type(time))
            #cols = np.where(theseClassStarts[j] < time < (theseClassStarts[j] + theseDurations[j]))
            cols = (theseClassStarts[j] < time) & (time < (theseClassStarts[j] + theseDurations[j]))
            # print(np.where(cols))
            classMagMatrix[i, cols] = theseMags[j]
            # exit()
    print('bar',np.where(cols))


    # # Apply a lowpass to round off the sharp start-stop edges of stimuli and octopamine:
    # lpParam = expParams.lpParam # default transition zone = 0.12 sec
    # L = round(lpParam/timeStep) # define the slope of low pass transitions here
    # lpWindow = hamming(L) # window of width L
    # lpWindow = lpWindow/sum(lpWindow)
    #
    # # window the stimulus time courses:
    # for i = 1:size(classMagMatrix,1)
    #   classMagMatrix(i,:) = conv(classMagMatrix(i,:), lpWindow,'same')
    # end
    #
    # # window the octopamine:
    # octoMag = expParams.octoMag
    # octoHits = zeros( 1, length(time))
    # octoStart = expParams.octoStart
    # durationOcto = expParams.durationOcto
    # octoStop = octoStart + durationOcto
    # for i = 1:length(octoStart)
    #   octoHits ( time >= octoStart(i) & time < octoStop(i)) = octoMag
    # end
    # octoHits = conv(octoHits,lpWindow,'same') # the low pass filter
    #
    # ## do SDE time-step evolution:
    #
    # # Use euler-maruyama SDE method, milstein's version.
    # #  Y (the vector of all neural firing rates) is structured as a row vector as follows: [ P, PI, L, K, E ]
    #
    # Po = 1*ones(nP, 1) # P are the normalized FRs of the excitatory PNs
    # PIo = 1*ones(nPI, 1) # PI are the normed FRs of the inhib PNs
    # Lo = 1*ones(nG, 1)
    # Ro = modelParams.Rspont
    # Ko = 1*ones(modelParams.nK, 1) # K are the normalized firing rates of the kenyon cells
    # Eo = 0*ones(modelParams.nE, 1) # start at zeros
    # initCond = vertcat( Po, PIo, Lo, Ro, Ko, Eo) # initial conditions for Y
    #
    # tspan = [ simStart,simStop ]
    # seedValue = 0 # to free up or fix randn. If = 0, a random seed value will be chosen. If > 0, the seed will be defined.
    #
    # # Run the SDE evolution:
    # thisRun= sdeEvolutionMnist_fn(tspan, initCond, time,...
    #   classMagMatrix, featureArray, octoHits, modelParams, expParams, seedValue )
    # # Time stepping is now done.
    #
    # ## Unpack Y and save results :
    # # Y is a matrix numTimePoints x nG. Each col is a PN, each row holds values for a single timestep
    # Y = thisRun.Y
    #
    # # save some inputs and outputs to a struct for argout:
    # simResults.T = thisRun.T # timing information
    # simResults.E =  thisRun.E
    # simResults.octoHits = octoHits
    # simResults.K2Efinal = thisRun.K2Efinal
    # simResults.P2Kfinal = thisRun.P2Kfinal
    #
    # if modelParams.saveAllNeuralTimecourses
    #   simResults.P = Y(:,1:nP)
    #   simResults.K = Y(:, nP + nPI + nG + nR + 1: nP + nPI + nG + nR + nK)
    # end
    # ## other neural timecourses:
    # #   simResults.PI = Y(:,nP + 1:nP + nPI)
    # #   simResults.L = Y(:, nP + nPI + 1:nP + nPI + nG)
    # #   simResults.R =  Y(:, nP + nPI + nG + 1: nP + nPI + nG + nR)

    return simResults
