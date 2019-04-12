def piecewiseLinearPseudoSigmoid(x, span, slope):
    '''
    Piecewise linear 'sigmoid' used for speed when squashing neural inputs in difference eqns
    '''
    import numpy as np

    y = x*slope
    y[y<(-span/2)] = -span/2 # replace values below -span/2
    y[y>(span/2)] = span/2 # replace values above span/2

    return y

def sdeEvoMNIST(tspan, initCond, time, classMagMatrix, featureArray,
    octoHits, mP, exP, seedValue):
    '''
    To include neural noise, evolve the differential equations using euler-
    maruyama, milstein version (see Higham's Algorithmic introduction to
    numerical simulation of SDE)
    Called by sdeWrapper. For use with MNIST experiments.
    Inputs:
        1. tspan: 1 x 2 vector = start and stop timepoints (sec)
        2. initCond: n x 1 vector = starting FRs for all neurons, order-specific
        3. time: vector of timepoints for stepping
        4. classMagMatrix: 10 x n matrix of stimulus magnitudes.
        Each row contains mags of digits from a given class.
        5. featureArray: numFeatures x numStimsPerClass x numClasses array
        6. octoHits: 1 x length(t) vector with octopamine strengths at each timepoint
        7. mP: modelParams, including connection matrices, learning rates, etc
        8. exP: experiment parameters with some timing info
        9. seedValue: for random number generation. 0 means start a new seed.
    Output:
        thisRun: object with attributes Y (vectors of all neural timecourses as rows);
                    T = t; and final mP.P2K and mP.K2E connection matrices.

    #---------------------------------------------------------------------------

    comment: for mnist, the book-keeping differs from the odor experiment set-up.
        Let nC = number of classes (1 - 10 for mnist).
        The class may change with each new digit, so there is be a counter
        that increments when stimMag changes from nonzero to zero.
        There are nC counters.

    inputs:
        1. tspan = 1 x 2 vector with start and stop times
        2. initCond = col vector with all starting values for P, L, etc
        3. time = start:step:stop; these are the time points for the evolution.
            Note we assume that noise and FRs have the same step size (based on Milstein's method)
        4. classMagMatrix = nC x N matrix where nC = # of different classes (for
            digits, up to 10), N = length(time = vector of time points). Each
            entry is the strength of a digit presentation.
        5. featureArray = mP.nF x kk x nC array, where mP.nF = numFeatures,
            kk >= number of puffs for that stim, and nC = # of classes.
        6. octoHits = 1 x N matrix. Each entry is a strength of octopamine
        7. mP = modelParams, a struct that contains values of all connectivity matrices,
            noise parameters, and timing params (eg when octo, stim and heb occur)
        8. exP = struct with timing params
        9. seedVal = starting seed value for reproducibility. optional arg
    outputs:
        1. T = m x 1 vector, timepoints used in evolution
        2. Y = m x K matrix, where K contains all FRs for P, L, PI, KC, etc; and
            each row is the FR at a given timepoint

    The function uses the noise params to create a Wiener process, then
    evolves the FR equations with the added noise

    Inside the difference equations we use a piecewise linear pseudo sigmoid,
    rather than a true sigmoid, for speed.

    Note re-calculating added noise:
        We want noise to be proportional to the mean spontFR of each neuron. So
        we need to get an estimate of this mean spont FR first. Noise is not
        added while neurons settle to initial SpontFR values.
        Then noise is added, proportional to spontFR. After this noise
        begins, meanSpontFRs converge to new values.
    So there is a 'stepped' system, as follows:
        1. no noise, neurons converge to initial meanSpontFRs = ms1
        2. noise proportional to ms1. neurons converge to new meanSpontFRs = ms2
        3. noise is proportional to ms2. neurons may converge to new
            meanSpontFRs = ms3, but noise is not changed. stdSpontFRs are
            calculated from ms3 time period.
    This has the following effects on simResults:
        1. In the heat maps and time-courses this will give a period of uniform FRs.
        2. The meanSpontFRs and stdSpontFRs are not 'settled' until after
        the exP.stopSpontMean3 timepoint.
    '''

    # needs to run before creating matplotlib figure(s)
    from support_functions.show_figs import getScreen
    getScreen()

    import numpy as np
    from scipy.special import erfinv
    import matplotlib.pyplot as plt

    # if argin seedValue is nonzero, fix the rand seed for reproducible results
    if seedValue:
        np.random.seed(seedValue)  # Reset random state

    spin = '/-\|' # create spinner for progress bar

    # numbers of objects
    (nC,_) = classMagMatrix.shape
    nP = mP.nG
    nL = mP.nG
    nR = mP.nG

    # DEV NOTE: Remove this next section (lines 89:94)
    ## noise in individual neuron FRs
    # These are vectors, one vector for each type:
    wRsig = mP.noiseRvec
    wPsig = mP.noisePvec
    wPIsig = mP.noisePIvec # no PIs for mnist
    wLsig = mP.noiseLvec
    wKsig = mP.noiseKvec
    wEsig = mP.noiseEvec

    # steady-state RN FR, base + noise:
    RspontRatios = mP.Rspont/mP.Rspont.mean() # used to scale stim inputs

    # DEV NOTE: Are we worried about the precision of these values (below)?
    ## param for sigmoid that squashes inputs to neurons:
    # the slope at x = 0 = mP.slopeParam*span/4
    kSlope = mP.slopeParam*mP.cK/4
    pSlope = mP.slopeParam*mP.cP/4
    piSlope = mP.slopeParam*mP.cPI/4 # no PIs for mnist
    rSlope = mP.slopeParam*mP.cR/4
    lSlope = mP.slopeParam*mP.cL/4

#-------------------------------------------------------------------------------

    dt = time[1] - time[0] # this is determined by start, stop and step in calling function
    N = int( (tspan[1] - tspan[0]) / dt ) # number of steps in noise evolution
    T = np.linspace(tspan[0], tspan[1], N) # the time vector

    def wiener(w_sig, meanSpont_, old_, tau_, inputs_):
        d_ = dt*(-old_*tau_ + inputs_)

        # Wiener noise:
        dW_ = np.sqrt(dt)*w_sig.squeeze()*meanSpont_*np.random.randn(*d_.shape)

        # combine them:
        return old_ + d_ + dW_

#-------------------------------------------------------------------------------

    P = np.zeros((nP, N))
    PI = np.zeros((mP.nPI, N)) # no PIs for mnist
    L = np.zeros((nL, N))
    R = np.zeros((nR, N))
    K = np.zeros((mP.nK, N))
    E = np.zeros((mP.nE, N))

    # initialize the FR matrices with initial conditions
    P[:,0] = initCond[ : nP ] # col vector
    PI[:,0] = initCond[ nP : nP + mP.nPI ] # no PIs for mnist
    L[:,0] = initCond[ nP + mP.nPI : nP + mP.nPI + nL ]
    R[:,0] = initCond[ nP + mP.nPI + nL : nP + mP.nPI + nL + nR ]
    K[:,0] = initCond[ nP + mP.nPI + nL + nR : nP + mP.nPI + nL + nR + mP.nK ]
    E[:,0] = initCond[ -mP.nE : ]
    # P2Kheb = mP.P2K # '-heb' suffix is used to show that it will vary with time
    # PI2Kheb = mP.PI2K # no PIs for mnist
    # K2Eheb = mP.K2E

    P2Kmask = mP.P2K > 0
    PI2Kmask = mP.PI2K > 0 # no PIs for mnist
    K2Emask = mP.K2E > 0
    newP2K = mP.P2K # initialize
    newPI2K = mP.PI2K # no PIs for mnist
    newK2E = mP.K2E

    # initialize the counters for the various classes
    classCounter = np.zeros(nC)

    # make a list of Ts for which heb is active
    hebRegion = np.zeros(T.shape)
    for i in range(len(exP.hebStarts)):
        inds = np.logical_and(T >= exP.hebStarts[i], T <= (exP.hebStarts[i] + exP.hebDurations[i]))
        hebRegion[inds] = 1

    ## DEBUG STEP:
    # fig, ax = plt.subplots()
    # ax.plot(T, hebRegion)
    # ax.set(title='hebRegion vs T')
    # ax.grid() # fig.savefig("test.png")
    # plt.show()

#-------------------------------------------------------------------------------

    meanCalc1Done = False # flag to prevent redundant calcs of mean spont FRs
    meanCalc2Done = False
    meanCalc3Done = False

    meanSpontP = np.zeros(nP)
    meanSpontPI = np.zeros(mP.nPI) # no PIs for mnist
    meanSpontL = np.zeros(nL)
    meanSpontR = np.zeros(nR)
    meanSpontK = np.zeros(mP.nK)
    # meanSpontE = np.zeros(mP.nE)
    # ssMeanSpontP = np.zeros(nP)
    # ssStdSpontP = np.ones(nP)

    maxSpontP2KtimesPval = 10 # placeholder until we have an estimate based on
                                # spontaneous PN firing rates

    ## Main evolution loop:
    # iterate through time steps to get the full evolution:
    for i in range(N): # i = index of the time point
        prog = int(15*(i/N))
        remain = 15-prog-1
        print(f"{spin[i%4]} SDE evolution:[{prog*'*'}{remain*' '}]", end='\r')

        # step = np.round(time[1] - time[0], 4)

        # DEV NOTE: Confused by this. What is purpose?
        if T[i]<(exP.stopSpontMean3 + 5) or mP.saveAllNeuralTimecourses:
            oldP = P[:,i]
            oldPI = PI[:,i] # no PIs for mnist
            oldL = L[:,i]
            oldR = R[:,i]
            oldK = K[:,i]
        else: # version to save memory:
            oldP = P.reshape(P.shape[0], -1)[:,-1]
            oldPI = PI.reshape(PI.shape[0], -1)[:,-1]
            oldL = L.reshape(L.shape[0], -1)[:,-1]
            oldR = R.reshape(R.shape[0], -1)[:,-1]
            oldK = K.reshape(K.shape[0], -1)[:,-1]
        oldE = E[:,i]
        oldT = T[i]

        oldP2K = newP2K # these are inherited from the previous iteration
        oldPI2K = newPI2K # no PIs for mnist
        oldK2E = newK2E

#-------------------------------------------------------------------------------

        # set flags to say:
        #   1. whether we are past the window where meanSpontFR is
        #       calculated, so noise should be weighted according to a first
        #       estimate of meanSpontFR (meanSpont1)
        #   2. whether we are past the window where meanSpontFR is recalculated
        #       to meanSpont2 and
        #   3. whether we are past the window where final stdSpontFR can be calculated.

        adjustNoiseFlag1 = oldT > exP.stopPreNoiseSpontMean1
        adjustNoiseFlag2 = oldT > exP.stopSpontMean2
        adjustNoiseFlag3 = oldT > exP.stopSpontMean3

        if adjustNoiseFlag1 and not(meanCalc1Done):
            # ie we have not yet calc'ed the noise weight vectors
            inds = np.nonzero(np.logical_and(T > exP.startPreNoiseSpontMean1,
                T < exP.stopPreNoiseSpontMean1))[0]
            meanSpontP = P[:,inds].mean(axis=1)
            meanSpontPI = PI[:,inds].mean(axis=1)
            meanSpontL = L[:,inds].mean(axis=1)
            meanSpontR = R[:,inds].mean(axis=1)
            meanSpontK = K[:,inds].mean(axis=1)
            # meanSpontE = E[:,inds].mean(axis=1)
            meanCalc1Done = 1 # so we don't calc this again


        if adjustNoiseFlag2 and not(meanCalc2Done):
            # ie we want to calc new noise weight vectors. This stage is surplus
            inds = np.nonzero(np.logical_and(T > exP.startSpontMean2,
                T < exP.stopSpontMean2))[0]
            meanSpontP = P[:,inds].mean(axis=1)
            meanSpontPI = PI[:,inds].mean(axis=1)
            meanSpontL = L[:,inds].mean(axis=1)
            meanSpontR = R[:,inds].mean(axis=1)
            meanSpontK = K[:,inds].mean(axis=1)
            # meanSpontE = E[:,inds].mean(axis=1)
            # stdSpontP = P[:,inds].std(axis=1) # for checking progress
            meanCalc2Done = 1 # so we don't calc this again

        if adjustNoiseFlag3 and not(meanCalc3Done):
            # we want to calc stdSpontP for use with LH channel and maybe for use in heb
            # maybe we should also use this for noise calcs (eg dWP). But the difference is slight.
            inds = np.nonzero(np.logical_and(T > exP.startSpontMean3,
                T < exP.stopSpontMean3))[0]
            ssMeanSpontP = P[:,inds].mean(axis=1) # 'ss' means steady state
            ssStdSpontP = P[:,inds].std(axis=1)
            ssMeanSpontPI = PI[:,inds].mean(axis=1) # no PIs for mnist
            ssStdSpontPI = PI[:,inds].std(axis=1) # no PIs for mnist
            meanCalc3Done = 1 # so we don't calc this again

            # set a minimum damping on KCs based on spontaneous PN activity,
            # sufficient to silence the MB silent absent odor:
            temp = np.sort(mP.P2K.dot(ssMeanSpontP)) # 'ascending' by default
            ignoreTopN = 1 # ie ignore this many of the highest vals
            temp = temp[:-ignoreTopN] # ignore the top few outlier K inputs
            maxSpontP2KtimesPval = temp.max() # The minimum global damping on the MB.
            print('maxSpontP2KtimesPval:', maxSpontP2KtimesPval)
            meanCalc3Done = 1

        # update classCounter
        if i: # if i is not zero
            for j in range(nC):
                if (classMagMatrix[j,i-1] == 0) and (classMagMatrix[j,i] > 0):
                    classCounter[j] += 1

        # get values of feature inputs at time index i, as a col vector.
        # This allows for simultaneous inputs by different classes, but current
        # experiments apply only one class at a time.
        thisInput = np.zeros(mP.nF)
        thisStimClassInd = []
        for j in range(nC):
            if classMagMatrix[j,i]: # if classMagMatrix[j,i] is not zero
                thisInput += classMagMatrix[j,i]*featureArray[:,int(classCounter[j]),j]
                thisStimClassInd.append(j)

#-------------------------------------------------------------------------------

        # get value at t for octopamine:
        thisOctoHit = octoHits[i]
        # octoHits is a vector with an octopamine magnitude for each time point

#-------------------------------------------------------------------------------

        # dR:
        # inputs: S = stim,  L = lateral neurons, mP.Rspont = spontaneous FR
        # NOTE: octo does not affect mP.Rspont. It affects R's response to input odors.
        Rinputs = (1 - thisOctoHit*mP.octo2R*mP.octoNegDiscount).squeeze()
        Rinputs[Rinputs<0] = 0 # pos. rectify Rinputs
        Rinputs *= -mP.L2R.dot(oldL)
        neur_act = mP.F2R.dot(thisInput)*RspontRatios.squeeze()
        neur_act = neur_act*(1 + thisOctoHit*mP.octo2R).squeeze()
        Rinputs = Rinputs + neur_act + mP.Rspont.squeeze()
        # DEV NOTE: Rinputs values seem higher than their ML version, even though
        # the inputs (octoMax, etc.) look the same - check if it's a problem.
        Rinputs = piecewiseLinearPseudoSigmoid(Rinputs, mP.cR, rSlope)

        # Wiener noise
        newR = wiener(wRsig, meanSpontR, oldR, mP.tauR, Rinputs)

#-------------------------------------------------------------------------------

        ## DEV NOTE: Check w/ CBD - dP, dPI, and dL all do the same thing at this point, correct?
        # dP:
        Pinputs = (1 - thisOctoHit*mP.octo2P*mP.octoNegDiscount).squeeze()
        Pinputs[Pinputs<0] = 0 # pos. rectify
        Pinputs *= -mP.L2P.dot(oldL)
        Pinputs += (mP.R2P.squeeze()*oldR)*(1 + thisOctoHit*mP.octo2P).squeeze()
        # ie octo increases responsivity to positive inputs and to spont firing, and
        # decreases (to a lesser degree) responsivity to neg inputs.
        Pinputs = piecewiseLinearPseudoSigmoid(Pinputs, mP.cP, pSlope)

        # Wiener noise
        newP = wiener(wPsig, meanSpontP, oldP, mP.tauP, Pinputs)

#-------------------------------------------------------------------------------

        # dPI: # no PIs for mnist
        PIinputs = (1 - thisOctoHit*mP.octo2PI*mP.octoNegDiscount).squeeze()
        PIinputs[PIinputs<0] = 0 # pos. rectify
        PIinputs *= -mP.L2PI.dot(oldL)
        PIinputs += mP.R2PI.dot(oldR)*(1 + thisOctoHit*mP.octo2PI).squeeze()
        # ie octo increases responsivity to positive inputs and to spont firing, and
        # decreases (to a lesser degree) responsivity to neg inputs.
        PIinputs = piecewiseLinearPseudoSigmoid(PIinputs, mP.cPI, piSlope)

        # Wiener noise
        newPI = wiener(wPIsig, meanSpontPI, oldPI, mP.tauPI, PIinputs)

#-------------------------------------------------------------------------------

        # dL:
        Linputs = (1 - thisOctoHit*mP.octo2L*mP.octoNegDiscount).squeeze()
        Linputs[Linputs<0] = 0 # pos. rectify
        Linputs *= -mP.L2L.dot(oldL)
        Linputs += (mP.R2L.squeeze()*oldR)*(1 + thisOctoHit*mP.octo2L).squeeze()
        Linputs = piecewiseLinearPseudoSigmoid(Linputs, mP.cL, lSlope)

        # Wiener noise
        newL = wiener(wLsig, meanSpontL, oldL, mP.tauL, Linputs)

#-------------------------------------------------------------------------------

        # Enforce sparsity on the KCs:
        # Global damping on KCs is controlled by mP.sparsityTarget
        # (during octopamine, by octSparsityTarget).
        # Assume that inputs to KCs form a gaussian, and use a threshold
        # calculated via std devs to enforce the correct sparsity.

        # Delays from AL -> MB and AL -> LH -> MB (~30 mSec) are ignored.

        # the # st devs to give the correct sparsity
        numNoOctoStds = np.sqrt(2)*erfinv(1 - 2*mP.sparsityTarget)
        numOctoStds = np.sqrt(2)*erfinv(1 - 2*mP.octoSparsityTarget)
        # select for either octo or no-octo
        numStds = (1-thisOctoHit)*numNoOctoStds + thisOctoHit*numOctoStds
        # set a minimum damping based on spontaneous PN activity, so that
        # the MB is silent absent odor
        minDamperVal = 1.2*maxSpontP2KtimesPval
        thisKinput = oldP2K.dot(oldP) - oldPI2K.dot(oldPI) # (no PIs for mnist, only Ps)

        # DEV NOTE: This value is different than the Matlab version -- clarify w/ CBD
        damper = thisKinput.mean() + numStds*thisKinput.std()
        damper = max(damper, minDamperVal)

        dampening = (damper*mP.kGlobalDampVec).squeeze() + oldPI2K.dot(oldPI)
        pos_octo = np.maximum(1 - mP.octo2K*thisOctoHit, 0).squeeze()

        Kinputs = oldP2K.dot(oldP)*(1 + thisOctoHit*mP.octo2K).squeeze() # but note that mP.octo2K == 0
        Kinputs -= dampening*pos_octo # but no PIs for mnist
        Kinputs = piecewiseLinearPseudoSigmoid(Kinputs, mP.cK, kSlope)

        # Wiener noise
        newK = wiener(wKsig, meanSpontK, oldK, mP.tauK, Kinputs)

#-------------------------------------------------------------------------------

        # Readout neurons E (EN = 'extrinsic neurons'):
        # These are readouts, so there is no sigmoid.
        # mP.octo2E == 0, since we are not stimulating ENs with octo.
        # dWE == 0 since we assume no noise in ENs.
        Einputs = oldK2E.dot(oldK)
        # oldK2E.dot(oldK)*(1 + thisOctoHit*mP.octo2E) # mP.octo2E == 0
        dE = dt*( -oldE*mP.tauE + Einputs )

        # Wiener noise
        dWE = 0 # noise = 0 => dWE == 0
        # combine them
        newE = oldE + dE + dWE # always non-neg

#-------------------------------------------------------------------------------

    ## HEBBIAN UPDATES:

        # Apply Hebbian learning to mP.P2K, mP.K2E:
        # For ease, use 'newK' and 'oldP', 'newE' and 'oldK', ie 1 timestep of delay.
        # We restrict hebbian growth in mP.K2E to connections into the EN of the
        # training stimulus.

        # Hebbian updates are active for about half the duration of each stimulus
        if True: #########REPLACE LATER!!!!!!!!! if hebRegion[i]:
            # the PN contribution to hebbian is based on raw FR
            tempP = oldP
            tempPI = oldPI # no PIs for mnist
            nonNegNewK = np.maximum(newK, 0) # since newK has not yet been made non-neg

            ## dP2K:
            dp2k = (1/mP.hebTauPK) * nonNegNewK.reshape(-1, 1).dot(tempP.reshape(-1, 1).T)
            dp2k *= P2Kmask #  if original synapse does not exist, it will never grow

            # decay some P2K connections if wished: (not used for mnist experiments)
            if mP.dieBackTauPK > 0:
                oldP2K -= oldP2K*(1/dieBackTauPK)*dt

            newP2K = np.maximum(oldP2K + dp2k, 0)
            newP2K = np.minimum(newP2K, mP.hebMaxPK)

#-------------------------------------------------------------------------------

            ## dPI2K: # no PIs for mnist
            dpi2k = (1/mP.hebTauPIK) * nonNegNewK.reshape(-1, 1).dot(tempPI.reshape(-1, 1).T)
            dpi2k *= PI2Kmask # if original synapse does not exist, it will never grow

            # kill small increases:
            temp = oldPI2K # this detour prevents dividing by zero
            temp[temp == 0] = 1
            keepMask = dpi2k/temp
            keepMask = keepMask.reshape(dpi2k.shape)
            dpi2k *= keepMask
            if mP.dieBackTauPIK > 0:
                oldPI2K -= oldPI2K*(1/dieBackTauPIK)*dt
            newPI2K = np.maximum(oldPI2K + dpi2k, 0)
            newPI2K = np.minimum(newPI2K, mP.hebMaxPIK)

#-------------------------------------------------------------------------------

            ## dK2E:
            tempK = oldK
            # oldK is already nonNeg
            dk2e = (1/mP.hebTauKE) * newE.reshape(-1, 1).dot(tempK.reshape(-1, 1).T)
            dk2e *= K2Emask

            # restrict changes to just the i'th row of mP.K2E, where i = ind of training stim
            restrictK2Emask = np.zeros(mP.K2E.shape)
            restrictK2Emask[thisStimClassInd,:] = 1
            dk2e *= restrictK2Emask

#-------------------------------------------------------------------------------

            # inactive connections for this EN die back:
            if mP.dieBackTauKE:
                # restrict dieBacks to only the trained EN
                targetMask = np.zeros(dk2e.shape)
                targetMask[ dk2e == 0 ] = 1
                targetMask *= restrictK2Emask
                dieBack = (oldK2E + 2)*(1/mP.dieBackTauKE)*dt
                # the '+1' allows weights to die to absolute 0
                oldK2E -= targetMask*dieBack

            newK2E = oldK2E + dk2e
            newK2E = np.maximum(newK2E, 0)
            newK2E = np.minimum(newK2E, mP.hebMaxKE)

        else: # case: no heb or no octo
            newP2K = oldP2K
            newPI2K = oldPI2K # no PIs for mnist
            newK2E = oldK2E

#-------------------------------------------------------------------------------

        # update the evolution matrices, disallowing negative FRs.
        if T[i]<(exP.stopSpontMean3 + 5) or mP.saveAllNeuralTimecourses:
            # case: do not save AL and MB neural timecourses after the noise calibration is done, to save on memory
            R[:,i] = np.maximum(newR, 0)
            P[:,i] = np.maximum(newP, 0)
            PI[:,i] = np.maximum(newPI, 0) # no PIs for mnist
            L[:,i] = np.maximum(newL, 0)
            K[:,i] = np.maximum(newK, 0)
            # DEV NOTE: delete following line. redundant with below
            # E[,i+1] = newE
        else:
            R = np.maximum(newR, 0)
            P = np.maximum(newP, 0)
            PI = np.maximum(newPI, 0) # no PIs for mnist
            L = np.maximum(newL, 0)
            K = np.maximum(newK, 0)

        E[:,i] = newE # always save full EN timecourses

    print()
    # Time-step simulation is now over.

    thisRun = dict() # pre-allocate
    ### FIGURE OUT WHY THE VARIABLE 'E' HAS A DIFFERENT SHAPE THAN THE OTHERS
    # # combine so that each row of fn output Y is a col of [P; PI; L; R; K]:
    if mP.saveAllNeuralTimecourses:
        Y = np.vstack((P, PI, L, R, K, E))
        thisRun['Y'] = Y.T.astype('float32') # convert to singles to save memory
    else:
        thisRun['Y'] = []

    thisRun['T'] = T.T.astype('float32') # store T as a col
    thisRun['E'] = E.T.astype('float32') # length(T) x mP.nE matrix
    thisRun['P2Kfinal'] = oldP2K.T.astype('float32')
    thisRun['K2Efinal'] = oldK2E.T.astype('float32')

    return thisRun
