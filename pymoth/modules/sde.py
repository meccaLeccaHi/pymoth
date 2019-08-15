#!/usr/bin/env python3

"""

.. module:: sde
   :platform: Unix
   :synopsis: Run stochastic differential equation simulation.

.. moduleauthor:: Adam P. Jones <ajones173@gmail.com>

"""
import os
import numpy as np
from scipy.special import erfinv
import matplotlib.pyplot as plt
from ..modules.show_figs import show_acc, show_timecourse

def sde_wrap( model_params, exp_params, feature_array ):
    """
    Runs the SDE time-stepped evolution of neural firing rates.

    4 steps:
        1. load various params needed for pre-evolution prep
        2. specify stim and octo courses
        3. interaction equations and step through simulation
        4. unpack evolution output and export

    Args:
        model_params (class): object with connection matrices, etc.
        exp_params (class): object with timing info about experiment, eg when stimuli are given.
        feature_array (numpy array): stimuli (numFeatures x numStimsPerClass x numClasses).

    Returns:
        sim_results (dict): EN timecourses and final P2K and K2E connection matrices.

    """

    ## 1. initialize states of various components:

    # unpack a few variables that are needed before the evolution stage:
    nP = model_params.nP # = nG
    nG = model_params.nG
    nPI = model_params.nPI
    nK = model_params.nK
    nR = model_params.nR # = nG
    nE = model_params.nE
    F2R = model_params.F2R

    ##  2b. Define Stimuli and Octopamine time courses:

    # set time span and events:
    sim_start = exp_params.sim_start
    sim_stop =  exp_params.sim_stop
    time_step = 2*0.01

    total_steps = (sim_stop - sim_start)/time_step
    time = np.linspace(sim_start, sim_stop-time_step, total_steps)

    class_labels = exp_params.class_labels
    # classMags = exp_params.classMags
    # create a class_mag_mat, each row giving the stimulus magnitudes of a different class:
    class_mag_mat = np.zeros((len(class_labels), len(time))) # ie 4 x len(time)
    for i,cl in enumerate(class_labels):
        # extract the relevant odor puffs. All vectors should be same size, in same order
        puffs = (exp_params.whichClass == cl)
        theseClassStarts = exp_params.stimStarts[puffs]
        theseDurations = exp_params.durations[puffs]
        theseMags = exp_params.classMags[puffs]

        for j in range(len(theseClassStarts)):
            cols = (theseClassStarts[j] < time) & (time < (theseClassStarts[j] + theseDurations[j]))
            class_mag_mat[i, cols] = theseMags[j]

    # Apply a lowpass to round off the sharp start-stop edges of stimuli and octopamine:
    # lpParam: default transition zone = 0.12 sec
    L = round(exp_params.lpParam/time_step) # define the slope of low pass transitions here
    lpWindow = np.hamming(L) # window of width L
    lpWindow /= lpWindow.sum()

    # window the stimulus time courses:
    for i in range(len(class_labels)):
        class_mag_mat[i,:] = np.convolve(class_mag_mat[i,:], lpWindow, 'same')

    # window the octopamine:
    # octoMag = exp_params.octoMag
    octo_hits = np.zeros(len(time))
    # octoStart = exp_params.octoStart
    # durationOcto = exp_params.durationOcto
    octoStop = [i + exp_params.durationOcto for i in exp_params.octoStart]
    for i in range(len(exp_params.octoStart)):
        hits = (time >= exp_params.octoStart[i]) & (time < octoStop[i])
        octo_hits[ hits ] = exp_params.octoMag
    octo_hits = np.convolve(octo_hits, lpWindow, 'same') # the low pass filter

    ## do SDE time-step evolution:

    # Use euler-maruyama SDE method, milstein's version.
    #  Y (the vector of all neural firing rates) is structured as a row vector as follows: [ P, PI, L, K, E ]
    Po = np.ones(nP) # P are the normalized FRs of the excitatory PNs
    PIo = np.ones(nPI) # PI are the normed FRs of the inhib PNs
    Lo = np.ones(nG)
    Ro = model_params.Rspont
    Ko = np.ones(model_params.nK) # K are the normalized firing rates of the Kenyon cells
    Eo = np.zeros(model_params.nE) # start at zeros
    init_cond = np.concatenate((Po, PIo, Lo, Ro, Ko, Eo) , axis=None) # initial conditions for Y

    tspan = ( sim_start, sim_stop )
    seed_val = 0 # to free up or fix randn
    # If = 0, a random seed value will be chosen. If > 0, the seed will be defined.

    # run the SDE evolution:
    this_run = sde_evo_mnist(tspan, init_cond, time, class_mag_mat, feature_array,
        octo_hits, model_params, exp_params, seed_val )
    # time stepping done

    ## Unpack Y and save results:
    # Y is a matrix numTimePoints x nG
    # Each col is a PN, each row holds values for a single timestep
    # Y = this_run['Y']

    # save some inputs and outputs to a struct for argout:
    sim_results = {
                    'T' : this_run['T'], # timing information
                    'E' : this_run['E'],
                    'octo_hits' : octo_hits,
                    'K2Efinal' : this_run['K2Efinal'],
                    'P2Kfinal' : this_run['P2Kfinal'],
                    'nE' : nE
                }

    return sim_results

def sde_evo_mnist(tspan, init_cond, time, class_mag_mat, feature_array,
    octo_hits, mP, exP, seed_val):
    """

    To include neural noise, evolve the differential equations using Euler-Maruyama, \
    Milstein version (see Higham's Algorithmic introduction to Numerical Simulation \
    of SDE).

    Called by :func:`sde_wrap`. For use with MNIST experiments.

    The function uses the noise params to create a Wiener process, then evolves \
    the FR equations with the added noise. Inside the difference equations we use \
    a piecewise linear pseudo sigmoid, rather than a true sigmoid, for speed.

    *Regarding re-calculating added noise:*
    We want noise to be proportional to the mean spontFR of each neuron. So \
    we need to get an estimate of this mean spont FR first. Noise is not \
    added while neurons settle to initial SpontFR values. Then noise is added, \
    proportional to spontFR. After this noise begins, mean_spont_FRs converge \
    to new values.

    *So, this is a 'stepped' system, that runs as follows:*
        #. no noise, neurons converge to initial mean_spont_FRs = ms1
        #. noise proportional to ms1. neurons converge to new mean_spont_FRs = ms2
        #. noise is proportional to ms2. neurons may converge to new `mean_spont_FRs` \
        = ms3, but noise is not changed. `std_spont_FRs` are calculated from ms3 \
        time period.

    *This has the following effects on simulation results:*
        #. In the heat maps and time-courses this will give a period of uniform FRs.
        #. The `mean_spont_FR`s and `std_spont_FR`s are not 'settled' until after \
        the `stopSpontMean3` timepoint.

    Args:
        tspan (tuple): start and stop timepoints (seconds)
        init_cond (numpy array): [n x 1] starting FRs for all neurons, order-specific
        time (numpy array): [start:step:stop] vector of timepoints for stepping \
        through the evolution. Note we assume that noise and FRs have the same step \
        size (based on Milstein's method).
        class_mag_mat (numpy array): [# of different classes X vector of time points] \
        each entry is the strength of a digit presentation.
        feature_array (numpy array): [numFeatures x numStimsPerClass x numClasses]
        octo_hits (numpy array): [1 x length(t)] octopamine strengths at each timepoint.
        mP (class): model_params, including connection matrices, learning rates, etc.
        exP (class): experiment parameters with some timing info.
        seed_val (int): optional arg for random number generation.

    Returns:
        this_run (dict):
            - T: [m x 1] timepoints used in evolution (timepoints used in evolution)
            - Y: [m x K] where K contains all FRs for P, L, PI, KC, etc; and each \
            row is the FR at a given timepoint
            - P2K: connection matrix
            - K2E: connection matrix

    """

    def piecewise_lin_pseudo_sig(x, span, slope):
        """
        Piecewise linear 'sigmoid' used for speed when squashing neural inputs in difference eqns.
        """
        y = x*slope
        y = np.maximum(y, -span/2) # replace values below -span/2
        y = np.minimum(y, span/2) # replace values above span/2
        return y

    def wiener(w_sig, mean_spont_, old_, tau_, inputs_):
        """
        Calculate wiener noise.
        """
        d_ = dt*(-old_*tau_ + inputs_)
        # Wiener noise:
        dW_ = np.sqrt(dt)*w_sig*mean_spont_*np.random.normal(0,1,(d_.shape))
        # combine them:
        return old_ + d_ + dW_

    # if argin seed_val is nonzero, fix the rand seed for reproducible results
    if seed_val:
        np.random.seed(seed_val)  # Reset random state

    spin = '/-\|' # create spinner for progress bar

    # numbers of objects
    (nC,_) = class_mag_mat.shape
    nP = mP.nG
    nL = mP.nG
    nR = mP.nG

    ## noise in individual neuron FRs
    # These are vectors, one vector for each type:
    wPsig = mP.noisePvec.squeeze()
    wPIsig = mP.noisePIvec.squeeze() # no PIs for mnist
    wLsig = mP.noiseLvec.squeeze()
    wRsig = mP.noiseRvec.squeeze()
    wKsig = mP.noiseKvec.squeeze()
    wEsig = mP.noiseEvec.squeeze()

    # steady-state RN FR, base + noise:
    RspontRatios = mP.Rspont/mP.Rspont.mean() # used to scale stim inputs

    ## param for sigmoid that squashes inputs to neurons:
    # the slope at x = 0 = mP.slope_param*span/4
    pSlope = mP.slope_param*mP.cP/4
    piSlope = mP.slope_param*mP.cPI/4 # no PIs for mnist
    lSlope = mP.slope_param*mP.cL/4
    rSlope = mP.slope_param*mP.cR/4
    kSlope = mP.slope_param*mP.cK/4

#-------------------------------------------------------------------------------

    dt = round(time[1] - time[0], 2) # this is determined by start, stop and step in calling function
    N = int( (tspan[1] - tspan[0]) / dt ) # number of steps in noise evolution
    T = np.linspace(tspan[0], tspan[1]-dt, N) # the time vector

#-------------------------------------------------------------------------------

    P = np.zeros((nP, N))
    PI = np.zeros((mP.nPI, N)) # no PIs for mnist
    L = np.zeros((nL, N))
    R = np.zeros((nR, N))
    K = np.zeros((mP.nK, N))
    E = np.zeros((mP.nE, N))

    # initialize the FR matrices with initial conditions
    P[:,0] = init_cond[ : nP ] # col vector
    PI[:,0] = init_cond[ nP : nP + mP.nPI ] # no PIs for mnist
    L[:,0] = init_cond[ nP + mP.nPI : nP + mP.nPI + nL ]
    R[:,0] = init_cond[ nP + mP.nPI + nL : nP + mP.nPI + nL + nR ]
    K[:,0] = init_cond[ nP + mP.nPI + nL + nR : nP + mP.nPI + nL + nR + mP.nK ]
    E[:,0] = init_cond[ -mP.nE : ]
    # P2Kheb = mP.P2K # '-heb' suffix is used to show that it will vary with time
    # PI2Kheb = mP.PI2K # no PIs for mnist
    # K2Eheb = mP.K2E

    P2Kmask = mP.P2K > 0
    PI2Kmask = mP.PI2K > 0 # no PIs for mnist
    K2Emask = mP.K2E > 0
    newP2K = mP.P2K.copy() # initialize
    newPI2K = mP.PI2K.copy() # no PIs for mnist
    newK2E = mP.K2E.copy()

    # initialize the counters for the various classes
    class_counter = np.zeros(nC)

    # make a list of Ts for which heb is active
    hebRegion = np.zeros(T.shape)
    for i in range(len(exP.hebStarts)):
        inds = np.bitwise_and(T >= exP.hebStarts[i], T <= (exP.hebStarts[i] + exP.hebDurations[i]))
        hebRegion[inds] = 1

    ## DEBUG STEP:
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots()
    # ax.plot(T, hebRegion)
    # ax.set(title='hebRegion vs T')
    # ax.grid() # fig.savefig("test.png")
    # plt.show()

#-------------------------------------------------------------------------------

    meanCalc1Done = False # flag to prevent redundant calcs of mean spont FRs
    meanCalc2Done = False
    meanCalc3Done = False

    mean_spont_P = np.zeros(nP)
    mean_spont_PI = np.zeros(mP.nPI) # no PIs for mnist
    mean_spont_L = np.zeros(nL)
    mean_spont_R = np.zeros(nR)
    mean_spont_K = np.zeros(mP.nK)
    # mean_spont_E = np.zeros(mP.nE)
    # ssMeanSpontP = np.zeros(nP)
    # ssStdSpontP = np.ones(nP)

    # placeholder until we have an estimate based on spontaneous PN firing rates
    maxSpontP2KtimesPval = 10

    ## Main evolution loop:
    # iterate through time steps to get the full evolution:
    for i in range(N-1): # i = index of the time point
        prog = int(15*(i/N))
        remain = 15-prog-1
        mult = 50 # multiplier (spinner speed control)
        print(f"{spin[int((i%(len(spin)*mult))/mult)]} SDE evolution:[{prog*'*'}{remain*' '}]", end='\r')

        # step = np.round(time[1] - time[0], 4)

        # if sufficiently early, or we want the entire evo
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

        oldP2K = newP2K.copy() # these are inherited from the previous iteration
        oldPI2K = newPI2K.copy() # no PIs for mnist
        oldK2E = newK2E.copy()

#-------------------------------------------------------------------------------

        # set flags to say:
        #   1. whether we are past the window where mean_spont_FR is calculated,
        #       so noise should be weighted according to a first estimate of
        #       mean_spont_FR (mean_spont_1)
        #   2. whether we are past the window where mean_spont_FR is recalculated
        #       to mean_spont_2 and
        #   3. whether we are past the window where final stdSpontFR can be calculated
        adjustNoiseFlag1 = oldT > exP.stopPreNoiseSpontMean1
        adjustNoiseFlag2 = oldT > exP.stopSpontMean2
        adjustNoiseFlag3 = oldT > exP.stopSpontMean3

        if adjustNoiseFlag1 and not(meanCalc1Done):
            # ie we have not yet calc'ed the noise weight vectors
            inds = np.nonzero(np.logical_and(T > exP.startPreNoiseSpontMean1,
                T < exP.stopPreNoiseSpontMean1))[0]
            mean_spont_P = P[:,inds].mean(axis=1)
            mean_spont_PI = PI[:,inds].mean(axis=1)
            mean_spont_L = L[:,inds].mean(axis=1)
            mean_spont_R = R[:,inds].mean(axis=1)
            mean_spont_K = K[:,inds].mean(axis=1)
            # mean_spont_E = E[:,inds].mean(axis=1)
            meanCalc1Done = 1 # so we don't calc this again

        if adjustNoiseFlag2 and not(meanCalc2Done):
            # ie we want to calc new noise weight vectors. This stage is surplus
            inds = np.nonzero(np.logical_and(T > exP.startSpontMean2,
                T < exP.stopSpontMean2))[0]
            mean_spont_P = P[:,inds].mean(axis=1)
            mean_spont_PI = PI[:,inds].mean(axis=1)
            mean_spont_L = L[:,inds].mean(axis=1)
            mean_spont_R = R[:,inds].mean(axis=1)
            mean_spont_K = K[:,inds].mean(axis=1)
            # mean_spont_E = E[:,inds].mean(axis=1)
            # stdSpontP = P[:,inds].std(axis=1) # for checking progress
            meanCalc2Done = 1 # so we don't calc this again

        if adjustNoiseFlag3 and not(meanCalc3Done):
            # we want to calc stdSpontP for use with LH channel and maybe for use in heb
            # maybe we should also use this for noise calcs (eg dWP).
            # But the difference is slight.
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
            maxSpontP2KtimesPval = temp.max() # The minimum global damping on the MB
            meanCalc3Done = 1

        # create class_counter - the counters for the various classes
        if i: # if i is not zero
            class_counter += np.logical_and(class_mag_mat[:,i-1]==0, class_mag_mat[:,i]>0)

        # get values of feature inputs at time index i, as a col vector.
        # This allows for simultaneous inputs by different classes, but current
        #   experiments apply only one class at a time.
        thisInput = np.zeros(mP.nF)
        thisStimClassInd = []
        for j in range(nC):
            if class_mag_mat[j,i]: # if class_mag_mat[j,i] is not zero
                # thisInput += class_mag_mat[j,i]*feature_array[:,int(class_counter[j]),j]
                imNum = int(class_counter[j] - 1) # indexing: need the '-1' so we don't run out of images
                thisInput += class_mag_mat[j,i]*feature_array[:,imNum,j]
                thisStimClassInd.append(j)

#-------------------------------------------------------------------------------

        # get value at t for octopamine:
        thisOctoHit = octo_hits[i]
        # octo_hits is a vector with an octopamine magnitude for each time point

#-------------------------------------------------------------------------------

        # dP:
        Pinputs = (1 - thisOctoHit*mP.octo2P*mP.octoNegDiscount).squeeze()
        Pinputs = np.maximum(Pinputs, 0) # pos. rectify
        Pinputs *= -mP.L2P.dot(oldL)
        Pinputs += (mP.R2P.squeeze()*oldR)*(1 + thisOctoHit*mP.octo2P).squeeze()
        # ie octo increases responsivity to positive inputs and to spont firing, and
        # decreases (to a lesser degree) responsivity to neg inputs.
        Pinputs = piecewise_lin_pseudo_sig(Pinputs, mP.cP, pSlope)

        # Wiener noise
        newP = wiener(wPsig, mean_spont_P, oldP, mP.tau_P, Pinputs)

#-------------------------------------------------------------------------------

        # dPI: # no PIs for mnist
        PIinputs = (1 - thisOctoHit*mP.octo2PI*mP.octoNegDiscount).squeeze()
        PIinputs = np.maximum(PIinputs, 0)  # pos. rectify
        PIinputs *= -mP.L2PI.dot(oldL)
        PIinputs += mP.R2PI.dot(oldR)*(1 + thisOctoHit*mP.octo2PI).squeeze()
        # ie octo increases responsivity to positive inputs and to spont firing, and
        # decreases (to a lesser degree) responsivity to neg inputs.
        PIinputs = piecewise_lin_pseudo_sig(PIinputs, mP.cPI, piSlope)

        # Wiener noise
        newPI = wiener(wPIsig, mean_spont_PI, oldPI, mP.tau_PI, PIinputs)

#-------------------------------------------------------------------------------

        # dL:
        Linputs = (1 - thisOctoHit*mP.octo2L*mP.octoNegDiscount).squeeze()
        Linputs = np.maximum(Linputs, 0) # pos. rectify
        Linputs *= -mP.L2L.dot(oldL)
        Linputs += (mP.R2L.squeeze()*oldR)*(1 + thisOctoHit*mP.octo2L).squeeze()
        Linputs = piecewise_lin_pseudo_sig(Linputs, mP.cL, lSlope)

        # Wiener noise
        newL = wiener(wLsig, mean_spont_L, oldL, mP.tau_L, Linputs)

#-------------------------------------------------------------------------------

        # dR:
        # inputs: S = stim,  L = lateral neurons, mP.Rspont = spontaneous FR
        # NOTE: octo does not affect mP.Rspont. It affects R's response to input odors.
        Rinputs = (1 - thisOctoHit*mP.octo2R*mP.octoNegDiscount).squeeze()
        Rinputs = np.maximum(Rinputs, 0) # pos. rectify Rinputs
        Rinputs *= -mP.L2R.dot(oldL)
        neur_act = mP.F2R.dot(thisInput)*RspontRatios.squeeze()
        neur_act *= (1 + thisOctoHit*mP.octo2R).squeeze()
        Rinputs += neur_act + mP.Rspont.squeeze()
        Rinputs = piecewise_lin_pseudo_sig(Rinputs, mP.cR, rSlope)

        # Wiener noise
        newR = wiener(wRsig, mean_spont_R, oldR, mP.tau_R, Rinputs)

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

        damper = thisKinput.mean() + numStds*thisKinput.std()
        damper = max(damper, minDamperVal)

        dampening = (damper*mP.kGlobalDampVec).squeeze() + oldPI2K.dot(oldPI)
        pos_octo = np.maximum(1 - mP.octo2K*thisOctoHit, 0).squeeze()

        Kinputs = oldP2K.dot(oldP)*(1 + thisOctoHit*mP.octo2K).squeeze() # but note that mP.octo2K == 0
        Kinputs -= dampening*pos_octo # but no PIs for mnist
        Kinputs = piecewise_lin_pseudo_sig(Kinputs, mP.cK, kSlope)

        # Wiener noise
        newK = wiener(wKsig, mean_spont_K, oldK, mP.tau_K, Kinputs)

#-------------------------------------------------------------------------------

        # Readout neurons E (EN = 'extrinsic neurons'):
        # These are readouts, so there is no sigmoid.
        # mP.octo2E == 0, since we are not stimulating ENs with octo.
        # dWE == 0 since we assume no noise in ENs.
        Einputs = oldK2E.dot(oldK)
        # oldK2E.dot(oldK)*(1 + thisOctoHit*mP.octo2E) # mP.octo2E == 0
        dE = dt*( -oldE*mP.tau_E + Einputs )

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
        if hebRegion[i]:
            # the PN contribution to hebbian is based on raw FR
            #tempP = oldP.copy()
            #tempPI = oldPI.copy() # no PIs for mnist
            nonNegNewK = np.maximum(newK, 0) # since newK has not yet been made non-neg

            ## dP2K:
            dp2k = (1/mP.heb_tau_PK) * nonNegNewK.reshape(-1, 1).dot(oldP.reshape(-1, 1).T)
            dp2k *= P2Kmask #  if original synapse does not exist, it will never grow

            # decay some P2K connections if wished: (not used for mnist experiments)
            if mP.die_back_tau_PK > 0:
                oldP2K *= -(1/mP.die_back_tau_PK)*dt

            newP2K = np.maximum(oldP2K + dp2k, 0)
            newP2K = np.minimum(newP2K, mP.hebMaxPK)

#-------------------------------------------------------------------------------

            ## dPI2K: # no PIs for mnist
            dpi2k = (1/mP.heb_tau_PIK) * nonNegNewK.reshape(-1, 1).dot(oldPI.reshape(-1, 1).T)
            dpi2k *= PI2Kmask # if original synapse does not exist, it will never grow

            # kill small increases:
            temp = oldPI2K.copy() # this detour prevents dividing by zero
            temp[temp == 0] = 1
            keepMask = dpi2k/temp
            keepMask = keepMask.reshape(dpi2k.shape)
            dpi2k *= keepMask
            if mP.die_back_tau_PIK > 0:
                oldPI2K -= oldPI2K*(1/die_back_tau_PIK)*dt
            newPI2K = np.maximum(oldPI2K + dpi2k, 0)
            newPI2K = np.minimum(newPI2K, mP.hebMaxPIK)

#-------------------------------------------------------------------------------

            ## dK2E:
            #tempK = oldK
            # oldK is already nonNeg
            dk2e = (1/mP.heb_tau_KE) * newE.reshape(-1, 1).dot(oldK.reshape(-1, 1).T)
            dk2e *= K2Emask

            # restrict changes to just the i'th row of mP.K2E, where i = ind of training stim
            restrictK2Emask = np.zeros(mP.K2E.shape)
            restrictK2Emask[thisStimClassInd,:] = 1
            dk2e *= restrictK2Emask

#-------------------------------------------------------------------------------

            # inactive connections for this EN die back:
            if mP.die_back_tau_KE:
                # restrict dieBacks to only the trained EN
                targetMask = np.zeros(dk2e.shape)
                targetMask[ dk2e == 0 ] = 1
                targetMask *= restrictK2Emask
                dieBack = (oldK2E + 2)*(1/mP.die_back_tau_KE)*dt
                # the '+1' allows weights to die to absolute 0
                oldK2E -= targetMask*dieBack

            newK2E = oldK2E + dk2e
            newK2E = np.maximum(newK2E, 0)
            newK2E = np.minimum(newK2E, mP.hebMaxKE)

        else: # case: no heb or no octo
            newP2K = oldP2K.copy()
            newPI2K = oldPI2K.copy() # no PIs for mnist
            newK2E = oldK2E.copy()

#-------------------------------------------------------------------------------

        # update the evolution matrices, disallowing negative FRs.
        if T[i]<(exP.stopSpontMean3 + 5) or mP.saveAllNeuralTimecourses:
            # case: do not save AL and MB neural timecourses after the noise
            #   calibration is done, to save on memory
            P[:,i+1] = np.maximum(newP, 0)
            PI[:,i+1] = np.maximum(newPI, 0) # no PIs for mnist
            L[:,i+1] = np.maximum(newL, 0)
            R[:,i+1] = np.maximum(newR, 0)
            K[:,i+1] = np.maximum(newK, 0)
        else:
            P = np.maximum(newP, 0)
            PI = np.maximum(newPI, 0) # no PIs for mnist
            L = np.maximum(newL, 0)
            R = np.maximum(newR, 0)
            K = np.maximum(newK, 0)

        E[:,i+1] = newE # always save full EN timecourses

    print('\r')
    # Time-step simulation is now over.

    this_run = dict() # pre-allocate
    # combine so that each row of fn output Y is a col of [P; PI; L; R; K]
    if mP.saveAllNeuralTimecourses:
        Y = np.vstack((P, PI, L, R, K, E))
        this_run['Y'] = Y.T
    else:
        this_run['Y'] = []

    this_run['T'] = T.T # store T as a col
    this_run['E'] = E.T # length(T) x mP.nE matrix
    this_run['P2Kfinal'] = oldP2K
    this_run['K2Efinal'] = oldK2E

    return this_run

def collect_stats(self, sim_results, exp_params, class_labels, show_time_plots,
    show_acc_plots, images_folder='', images_filename='', screen_size=(1920,1080)):
    """
    *Collect stats on readout neurons (EN).*
    Collect stats (median, mean, and std of FR) for each digit, pre- and post-training. \
    Digits are referred to as odors, or as odor puffs.

    Args:
        sim_results (dict): simulation results (output from :func:`sde_wrap`)
        exp_params (class): timing info about experiment, eg when stimuli are given
        class_labels (numpy array): labels, eg 0:9 for MNIST
        show_time_plots (bool): show EN timecourses
        show_acc_plots (bool): show changes in accuracy
        images_filename (str): [optional] to generate image filenames when saving
        images_folder (str): [optional] directory to save results
        screen_size (tuple): [optional] screen size (width, height) for images

    Returns
    -------
        results (dict)
            pre_mean_resp (numpy array)
                [numENs x numOdors] mean of EN responses pre-training
            pre_std_resp (numpy array)
                [numENs x numOdors] std of EN responses pre-training
            post_mean_resp (numpy array)
                [numENs x numOdors] mean of EN responses post-training
            post_std_resp (numpy array)
                [numENs x numOdors] std of EN responses post-training
            percent_change_mean_resp (numpy array)
                [1 x numOdors]
            trained (list)
                indices corresponding to the odor(s) that were trained
            pre_spont_mean (float)
                mean of pre_spont
            pre_spont_std (float)
                std of pre_spont
            post_spont_mean (float)
                mean of post_spont
            post_spont_std (float)
                std of post_spont
    """

    # concurrent octopamine
    if sim_results['octo_hits'].max() > 0:
        octo_times = sim_results['T'][ sim_results['octo_hits'] > 0 ]
    else:
        octo_times = []

    # calc spont stats
    pre_spont = sim_results['E'][ np.logical_and(exp_params.preHebSpontStart < sim_results['T'],
                                    sim_results['T'] < exp_params.preHebSpontStop) ]
    post_spont = sim_results['E'][ np.logical_and(exp_params.postHebSpontStart < sim_results['T'],
                                    sim_results['T'] < exp_params.postHebSpontStop) ]

    pre_heb_mean = pre_spont.mean()
    pre_heb_std = pre_spont.std()
    post_heb_mean = post_spont.mean()
    post_heb_std = post_spont.std()

    ## Set regions to examine:
    # 1. data from exp_params
    # stim_starts = exp_params.stim_starts # get time-steps from very start of sim
    stim_starts = exp_params.stimStarts*(exp_params.classMags > 0) # ie only use non-zero puffs
    which_class = exp_params.whichClass*(exp_params.classMags > 0)
    class_labels = np.unique(which_class)

    # pre-allocate list of empty dicts
    results = [dict() for i in range(sim_results['nE'])]

    # make one stats plot per EN. Loop through ENs:
    for en_ind in range(sim_results['nE']):

        en_resp = sim_results['E'][:, en_ind]

        ## calculate pre- and post-train odor response stats
        # assumes that there is at least 1 sec on either side of an odor without octo

        # pre-allocate for loop
        pre_train_resp = np.full(len(stim_starts), np.nan)
        post_train_resp = np.full(len(stim_starts), np.nan)

        for i, t in enumerate(stim_starts):
            # Note: to find no-octo stim_starts, there is a certain amount of machinery
            # in order to mesh with the timing data from the experiment.
            # For some reason octo_times are not recorded exactly as listed in format
            # short mode. So we need to use abs difference > small thresh, rather
            # than ~ismember(t, octo_times):
            small = 1e-8 # .00000001
            # assign no-octo, PRE-train response val (or -1)
            pre_train_resp[i] = -1 # as flag
            if (len(octo_times)==0) or ((abs(octo_times - t).min() > small) and (t < exp_params.startTrain)):
                resp_ind = np.logical_and(t-1 < sim_results['T'], sim_results['T'] < t+1)
                pre_train_resp[i] = en_resp[resp_ind].max()

            # assign no-octo, POST-train response val (or -1)
            post_train_resp[i] = -1
            if len(octo_times)!=0:
                if (abs(octo_times - t).min() > small) and (t > exp_params.endTrain):
                    resp_ind = np.logical_and(t-1 < sim_results['T'], sim_results['T'] < t+1)
                    post_train_resp[i] = en_resp[resp_ind].max()

        # pre-allocate for loop
        pre_mean_resp, pre_median_resp, pre_std_resp, pre_num_puffs, post_mean_resp, \
            post_median_resp, post_std_resp, post_num_puffs = \
            [np.full(len(class_labels), np.nan) for _ in range(8)]

        # calc no-octo stats for each odor, pre and post train:
        for k, cl in enumerate(class_labels):
            current_class = which_class==cl
            pre_SA = pre_train_resp[np.logical_and(pre_train_resp>=0, current_class)]
            post_SA = post_train_resp[np.logical_and(post_train_resp>=0, current_class)]

            ## calculate the averaged sniffs of each sample: SA means 'sniffsAveraged'
            # this will contain the average responses over all sniffs for each sample
            if len(pre_SA)==0:
                pre_mean_resp[k] = -1
                pre_median_resp[k] = -1
                pre_std_resp[k] = -1
                pre_num_puffs[k] = 0
            else:
                pre_mean_resp[k] = pre_SA.mean()
                pre_median_resp[k] = np.median(pre_SA)
                pre_std_resp[k] = pre_SA.std()
                pre_num_puffs[k] = len(pre_SA)

            if len(post_SA)==0:
                post_mean_resp[k] = -1
                post_median_resp[k] = -1
                post_std_resp[k] = -1
                post_num_puffs[k] = 0
            else:
                post_mean_resp[k] = post_SA.mean()
                post_median_resp[k] = np.median(post_SA)
                post_std_resp[k] = post_SA.std()
                post_num_puffs[k] = len(post_SA)

        # # to plot +/- 1 std of % change in mean_resp, we want the std of our
        # # estimate of the mean = std_resp/sqrt(numPuffs). Make this calc:
        # pre_std_mean_est = pre_std_resp/np.sqrt(pre_num_puffs)
        # post_std_mean_est = post_std_resp/np.sqrt(post_num_puffs)

        pre_SA = np.nonzero(pre_num_puffs > 0)[0]
        post_SA = np.nonzero(post_num_puffs > 0)[0]
        post_offset = post_SA + 0.25

        percent_change_mean_resp = (100*(post_mean_resp[pre_SA] - pre_mean_resp[pre_SA]))\
                                    /pre_mean_resp[pre_SA]
        percent_change_noise_sub_mean_resp = \
                                (100*(post_mean_resp[pre_SA] - pre_mean_resp[pre_SA] - post_heb_mean))\
                                /pre_mean_resp[pre_SA]

        percent_change_median_resp = (100*(post_median_resp[pre_SA] - pre_median_resp[pre_SA]))\
                                /pre_median_resp[pre_SA]
        percent_change_noise_sub_median_resp = \
                                (100*(post_median_resp[pre_SA] - pre_median_resp[pre_SA] - post_heb_mean))\
                                /pre_median_resp[pre_SA]

        # plot stats (if selected)
        if show_acc_plots:
            fig = show_acc(pre_SA, post_SA, en_ind, pre_mean_resp, pre_median_resp,
                pre_std_resp, post_offset, post_mean_resp, post_median_resp, post_std_resp,
                class_labels, pre_heb_mean, pre_heb_std, post_heb_mean, post_heb_std,
                percent_change_mean_resp, screen_size)

            # create directory for images (if doesnt exist)
            if images_filename and not os.path.isdir(images_folder):
                os.mkdir(images_folder)
                print('Creating results directory: {}'.format(images_folder))
            # save fig
            fig_name = images_folder + os.sep + images_filename + '_en{}.png'.format(en_ind)
            fig.savefig(fig_name, dpi=100)
            print(f'Figure saved: {fig_name}')

        #-----------------------------------------------------------------------

        # store results in a list of dicts
        results[en_ind]['pre_train_resp'] = pre_train_resp # preserves all the sniffs for each stimulus
        results[en_ind]['post_train_resp'] = post_train_resp
        results[en_ind]['pre_resp_sniffs_ave'] = pre_SA # the averaged sniffs for each stimulus
        results[en_ind]['post_resp_sniffs_ave'] = post_SA
        results[en_ind]['odor_class'] = which_class
        results[en_ind]['percent_change_mean_resp'] = percent_change_mean_resp # key stat
        results[en_ind]['percent_change_noise_sub_mean_resp'] = percent_change_noise_sub_mean_resp
        results[en_ind]['rel_change_noise_sub_mean_resp'] = \
                percent_change_noise_sub_mean_resp / percent_change_noise_sub_mean_resp[en_ind]
        results[en_ind]['percent_change_median_resp'] = percent_change_median_resp
        results[en_ind]['percent_change_noise_sub_median_resp'] = percent_change_noise_sub_median_resp
        results[en_ind]['rel_change_noise_sub_median_resp'] = \
                ( (post_median_resp - pre_median_resp - post_heb_mean )/pre_median_resp ) / \
                ( (post_median_resp[en_ind] - pre_median_resp[en_ind] - post_heb_mean )/pre_median_resp[en_ind] )
        results[en_ind]['trained'] = en_ind
        # EN odor responses, pre and post training.
        # these should be vectors of length numStims
        results[en_ind]['pre_mean_resp'] = pre_mean_resp
        results[en_ind]['pre_std_resp'] = pre_std_resp
        results[en_ind]['post_mean_resp'] = post_mean_resp
        results[en_ind]['post_std_resp'] = post_std_resp
        # spont responses, pre and post training
        results[en_ind]['pre_spont_mean'] = pre_spont.mean()
        results[en_ind]['pre_spont_std'] = pre_spont.std()
        results[en_ind]['post_spont_mean'] = post_spont.mean()
        results[en_ind]['post_spont_std'] = post_spont.std()

    ## Plot EN timecourses normalized by mean digit response
    if show_time_plots:

        # go through each EN
        for en_ind in range(sim_results['nE']): # recal EN1 targets digit class 1, EN2 targets digit class 2, etc

            if en_ind%3 == 0:
                # make a new figure at ENs 4, 7, 10
                fig_sz = [np.floor(i/100) for i in screen_size]
                fig = plt.figure(figsize=fig_sz, dpi=100)

            ax = fig.add_subplot(3, 1, (en_ind%3)+1)
            show_timecourse(ax, en_ind, sim_results, octo_times, class_labels, results,
                exp_params, stim_starts, which_class )

            # Save EN timecourse:
            if os.path.isdir(images_folder) and \
            (en_ind%3 == 2 or en_ind == (sim_results['nE']-1)):
                fig_name = images_folder + os.sep + images_filename + '_en_timecourses{}.png'.format(en_ind)
                fig.savefig(fig_name, dpi=100)
                print(f'Figure saved: {fig_name}')

    return results

# MIT license:
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial
# portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
# PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN
# AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
