def sdeEvoMNIST(tspan, initCond, time, classMagMatrix, featureArray,
    octoHits, mP, exP, seedValue):
    # To include neural noise, evolve the differential equations using euler-
    # maruyama, milstein version (see Higham's Algorithmic introduction to
    # numerical simulation of SDE)
    # Called by sdeWrapper. For use with MNIST experiments.
    # Inputs:
    #   1. tspan: 1 x 2 vector = start and stop timepoints (sec)
    #   2. initCond: n x 1 vector = starting FRs for all neurons, order-specific
    #   3. time: vector of timepoints for stepping
    #   4. classMagMatrix: 10 x n matrix of stimulus magnitudes.
    #      Each row contains mags of digits from a given class
    #   5. featureArray: numFeatures x numStimsPerClass x numClasses array
    #   6. octoHits: 1 x length(t) vector with octopamine strengths at each timepoint
    #   7. mP: modelParams, including connection matrices, learning rates, etc
    #   8. exP: experiment parameters with some timing info
    #   9. seedValue: for random number generation. 0 means start a new seed.
    # Output:
    #   thisRun: object with attributes Y (vectors of all neural timecourses as rows); T = t;
    #                 and final mP.P2K and mP.K2E connection matrices.

#-------------------------------------------------------------------------------

    # comment: for mnist, the book-keeping differs from the odor experiment set-up.
    #           Let nC = number of classes (1 - 10 for mnist).
    #           The class may change with each new digit, so there is
    #           be a counter that increments when stimMag changes from nonzero
    #           to zero. there are nC counters.

    # inputs:
    #       1. tspan = 1 x 2 vector with start and stop times
    #       2. initCond = col vector with all starting values for P, L, etc
    #       3. time = start:step:stop; these are the time points for the evolution.
    #          Note we assume that noise and FRs have the same step size (based on Milstein's method)
    #       4. classMagMatrix = nC x N matrix where nC = # different classes (for digits, up to 10), N = length(time =
    #          vector of time points). Each entry is the strength of a digit presentation.
    #       5. featureArray = mP.nF x kk x nC array, where mP.nF = numFeatures, kk >= number of
    #           puffs for that stim, and c = # classes.
    #       6. octoHits = 1 x N matrix. Each entry is a strength of octopamine
    #       7. mP = modelParams, a struct that contains values of all connectivity matrices, noise
    #            parameters, and timing params (eg when octo, stim and heb occur)
    #       8. exP = struct with timing params
    #       9. seedVal = starting seed value for reproducibility. optional arg
    # outputs:
    #       1. T = m x 1 vector, timepoints used in evolution
    #       2. Y = m x K matrix, where K contains all FRs for P, L, PI, KC, etc; and
    #                  each row is the FR at a given timepoint

    # The function uses the noise params to create a Wiener process, then
    # evolves the FR equations with the added noise

    # Inside the difference equations we use a piecewise linear pseudo sigmoid,
    # rather than a true sigmoid, for speed.

    # Note re-calculating added noise:
    #   We want noise to be proportional to the mean spontFR of each neuron. So
    #   we need to get an estimate of this mean spont FR first. Noise is not
    #   added while neurons settle to initial SpontFR
    #   values. Then noise is added, proportional to spontFR. After this  noise
    #   begins, meanSpontFRs converge to new values.
    #  So there is a 'stepped' system, as follows:
    #       1. no noise, neurons converge to initial meanSpontFRs = ms1
    #       2. noise proportional to ms1. neurons converge to new meanSpontFRs = ms2
    #       3. noise is proportional to ms2. neurons may converge to new
    #          meanSpontFRs = ms3, but noise is not changed. stdSpontFRs are
    #          calculated from ms3 time period.
    #   This has the following effects on simResults:
    #       1. In the heat maps and time-courses this will give a period of uniform FRs.
    #       2. The meanSpontFRs and stdSpontFRs are not 'settled' until after
    #          the stopSpontMean3 timepoint.

#-------------------------------------------------------------------------------

    import numpy as np

    # if argin seedValue is nonzero, fix the rand seed for reproducible results
    if seedValue:
        np.random.seed(seedValue)  # Reset random state

    # numbers of objects
    (nC,_) = classMagMatrix.shape
    print('nC:',nC)
    quit()

    # numbers of objects
    # nC = size(classMagMatrix,1)
    # nP = mP.nG
    # nL = mP.nG
    # nR = mP.nG

    # # noise in individual neuron FRs. These are vectors, one vector for each type:
    # wRsig = mP.noiseRvec
    # wPsig = mP.noisePvec
    # wPIsig = mP.noisePIvec # no PIs for mnist
    # wLsig = mP.noiseLvec
    # wKsig = mP.noiseKvec
    # wEsig = mP.noiseEvec

    # kGlobalDampVec = mP.kGlobalDampVec # uniform 1's currently, ie LH inhibition hits all KCs equally

    # # steady-state RN FR, base + noise:
    # Rspont = mP.Rspont
    # RspontRatios = Rspont/mean(Rspont) # used to scale stim inputs

    # # param for sigmoid that squashes inputs to neurons:
    # slopeParam = mP.slopeParam # slope of sigmoid at 0 = slopeParam*c/4, where c = mP.cR, mP.cP, mP.cL, etc
    # # the slope at x = 0 = slopeParam*span/4
    # kSlope = slopeParam*mP.cK/4
    # pSlope = slopeParam*mP.cP/4
    # piSlope = slopeParam*mP.cPI/4 # no PIs for mnist
    # rSlope = slopeParam*mP.cR/4
    # lSlope = slopeParam*mP.cL/4

    # # end timepoints for the section used to define mean spontaneous firing rates, in order to calibrate noise.
    # # To let the system settle, we recalibrate noise levels to current spontaneous FRs in stages.
    # # This ensures that in steady state, noise levels are correct in relation to mean FRs.
    # startPreNoiseSpontMean1 = exP.startPreNoiseSpontMean1
    # stopPreNoiseSpontMean1 = exP.stopPreNoiseSpontMean1
    # startSpontMean2 = exP.startSpontMean2
    # stopSpontMean2 = exP.stopSpontMean2
    # startSpontMean3 = exP.startSpontMean3
    # stopSpontMean3 = exP.stopSpontMean3

#-------------------------------------------------------------------------------

    # dt = time(2) - time(1) # this is determined by start, stop and step in calling function
    # N = floor( (tspan(2) - tspan(1)) / dt ) # number of steps in noise evolution
    # T(1:N) = tspan(1):dt:tspan(2)-dt # the time vector

#-------------------------------------------------------------------------------

    # P = zeros(nP,N)
    # PI = zeros(mP.nPI,N) # no PIs for mnist
    # L = zeros(nL,N)
    # R = zeros(nR, N)
    # K = zeros(mP.nK, N)
    # E = zeros(mP.nE, N)

    # # initialize the FR matrices with initial conditions:
    # P(:,1) = initCond( 1 : nP) # col vector
    # PI(:,1) = initCond( nP + 1 : nP + mP.nPI) # no PIs for mnist
    # L(:,1) = initCond( nP + mP.nPI + 1 : nP + mP.nPI + nL )
    # R(:,1) = initCond(nP + mP.nPI + nL + 1: nP + mP.nPI + nL + nR)
    # K(:,1) = initCond(nP + mP.nPI + nL + nR + 1: nP + mP.nPI + nL + nR + mP.nK)
    # E(:,1) = initCond(end - mP.nE + 1 : end)
    # P2Kheb{1} = mP.P2K # '-heb' suffix is used to show that it will vary with time
    # PI2Kheb{1} = mP.PI2K # no PIs for mnist
    # K2Eheb{1} = mP.K2E
    # P2Kmask = mP.P2K > 0
    # PI2Kmask = mP.PI2K > 0 # no PIs for mnist
    # K2Emask = mP.K2E > 0
    # newP2K = mP.P2K # initialize
    # newPI2K = mP.PI2K # no PIs for mnist
    # newK2E = mP.K2E

    ## initialize the counters for the various classes:
    # classCounter = zeros(size(classMagMatrix,1), 1)

    ## make a list of Ts for which heb is active:
    # hebRegion = zeros(size(T))
    # for i = 1:length(exP.hebStarts)
    #     hebRegion(T >= exP.hebStarts(i) & T <= exP.hebStarts(i) + exP.hebDurations(i) ) = 1
    # end

    ## DEBUG STEP:
    # # figure, plot(T, hebRegion), title('hebRegion vs T')

#-------------------------------------------------------------------------------

    # meanCalc1Done = False # flag to prevent redundant calcs of mean spont FRs
    # meanCalc2Done = False
    # meanCalc3Done = False
    # meanSpontR = 0*ones(size(R(:,1)))
    # meanSpontP = 0*ones(size(P(:,1)))
    # meanSpontPI = 0*ones(size(PI(:,1))) # no PIs for mnist
    # meanSpontL = 0*ones(size(L(:,1)))
    # meanSpontK = 0*ones(size(K(:,1)))
    # meanSpontE = 0*ones(size(E(:,1)))
    # ssMeanSpontP = 0*ones(size(P(:,1)))
    # ssStdSpontP = ones(size(P(:,1)))

    # maxSpontP2KtimesPval = 10 # placeholder until we have an estimate based on spontaneous PN firing rates
    # # The main evolution loop:
    # # iterate through time steps to get the full evolution:
    # for i = 1:N-1        # i = index of the time point

    #     step = time(2) - time(1)

    #     if T(i) < stopSpontMean3 + 5 || mP.saveAllNeuralTimecourses
    #         oldR = R(:,i)
    #         oldP = P(:,i)
    #         oldPI = PI(:,i) # no PIs for mnist
    #         oldL = L(:,i)
    #         oldK = K(:,i)
    #     else    # version to save memory:
    #         oldR = R(:,end)
    #         oldP = P(:,end)
    #         oldPI = PI(:,end)
    #         oldL = L(:,end)
    #         oldK = K(:,end)
    #     end
    #     oldE = E(:,i)
    #     oldT = T(i)

    #     oldP2K = newP2K # these are inherited from the previous iteration
    #     oldPI2K = newPI2K # no PIs for mnist
    #     oldK2E = newK2E

#-------------------------------------------------------------------------------

    #     # set flags to say:
    #     #   1. whether we are past the window where meanSpontFR is
    #     #       calculated, so noise should be weighted according to a first
    #     #       estimate of meanSpontFR (meanSpont1)
    #     #   2. whether we are past the window where meanSpontFR is recalculated to meanSpont2 and
    #     #   3. whether we are past the window where final stdSpontFR can be calculated.

    #     adjustNoiseFlag1 = oldT > stopPreNoiseSpontMean1
    #     adjustNoiseFlag2 = oldT > stopSpontMean2
    #     adjustNoiseFlag3 = oldT > stopSpontMean3

    #     if adjustNoiseFlag1 && ~meanCalc1Done  # ie we have not yet calc'ed
    #   the noise weight vectors:
    #         inds = find(T > startPreNoiseSpontMean1 & T < stopPreNoiseSpontMean1)
    #         meanSpontP = mean(P(:,inds),2)
    #         meanSpontR = mean(R(:,inds),2)
    #         meanSpontPI = mean(PI(:,inds),2)
    #         meanSpontL = mean(L(:,inds),2)
    #         meanSpontK = mean(K(:,inds), 2)
    #         meanSpontE = mean(E(:,inds), 2 )
    #         meanCalc1Done = 1 # so we don't calc this again
    #     end
    #     if adjustNoiseFlag2 && ~meanCalc2Done  # ie we want to calc new noise weight vectors. This stage is surplus.
    #         inds = find(T > startSpontMean2 & T < stopSpontMean2)
    #         meanSpontP = mean(P(:,inds),2)
    #         meanSpontR = mean(R(:,inds),2)
    #         meanSpontPI = mean(PI(:,inds),2)
    #         meanSpontL = mean(L(:,inds),2)
    #         meanSpontK = mean(K(:,inds), 2)
    #         meanSpontE = mean(E(:,inds), 2)
    #         stdSpontP = std(P(:,inds),0, 2) # for checking progress
    #         meanCalc2Done = 1
    #     end
    #     if adjustNoiseFlag3 && ~meanCalc3Done  # we want to calc stdSpontP for use with LH channel and maybe for use in heb:
    #         # maybe we should also use this for noise calcs (eg dWP). But the difference is slight.
    #         inds = find(T > startSpontMean3 & T < stopSpontMean3)
    #         ssMeanSpontP = mean(P(:,inds),2) # 'ss' means steady state
    #         ssStdSpontP = std(P(:,inds),0, 2)
    #         ssMeanSpontPI = mean(PI(:,inds),2) # no PIs for mnist
    #         ssStdSpontPI = std(PI(:,inds),0, 2) # no PIs for mnist
    #         meanCalc3Done = 1
    #         # set a minimum damping on KCs based on spontaneous PN activity, sufficient to silence the MB silent absent odor:
    #         temp = mP.P2K*ssMeanSpontP
    #         temp = sort(temp,'ascend')
    #         ignoreTopN = 1 # ie ignore this many of the highest vals
    #         temp = temp(1:end - ignoreTopN) # ignore the top few outlier K inputs.
    #         maxSpontP2KtimesPval = max(temp) # The minimum global damping on the MB.
    #         meanCalc3Done = 1
    #     end

    #     # update classCounter:
    #     if i > 1
    #         for j = 1:nC
    #             if classMagMatrix(j,i-1) == 0 && classMagMatrix(j,i) > 0
    #                 classCounter(j) = classCounter(j) + 1
    #             end
    #         end
    #     end

    #     # get values of feature inputs at time index i, as a col vector.
    #     # This allows for simultaneous inputs by different classes, but current
    #     # experiments apply only one class at a time.
    #     thisInput = zeros(mP.nF,1)
    #     thisStimClassInd = []
    #     for j = 1:nC
    #         if classMagMatrix(j,i) > 0
    #             thisInput = thisInput + classMagMatrix(j,i)*featureArray(:, classCounter(j), j)
    #             thisStimClassInd = [ thisStimClassInd, j ]
    #         end
    #     end

#-------------------------------------------------------------------------------

    #     # get value at t for octopamine:
    #     thisOctoHit = octoHits(i) # octoHits is a vector with an octopamine magnitude for each time point.

#-------------------------------------------------------------------------------

    #     # dR:
    #     # inputs: S = stim,  L = lateral neurons, Rspont = spontaneous FR
    #     # NOTE: octo does not affect Rspont. It affects R's response to input odors.
    #     Rinputs = -mP.L2R*oldL.*max( 0, (ones(mP.nG,1) - thisOctoHit*mP.octo2R*mP.octoNegDiscount ) )   + ...
    #         (mP.F2R*thisInput).*RspontRatios.*( ones(mP.nG,1) + thisOctoHit*mP.octo2R ) + Rspont

    #     Rinputs = piecewiseLinearPseudoSigmoid_fn (Rinputs, mP.cR, rSlope)

    #     dR = dt*( -oldR*mP.tauR + Rinputs )

#-------------------------------------------------------------------------------

    #     # Wiener noise:
    #     dWR = sqrt(dt)*wRsig.*meanSpontR.*randn(size(dR))
    #     # combine them:
    #     newR = oldR + dR + dWR

#-------------------------------------------------------------------------------

    #     # dP:
    #     Pinputs = -mP.L2P*oldL.*max( 0, (1 - thisOctoHit*mP.octo2P*mP.octoNegDiscount) ) + (mP.R2P.*oldR).*(1 + thisOctoHit*mP.octo2P)
    #     # ie octo increases responsivity to positive inputs and to spont firing, and
    #     # decreases (to a lesser degree) responsivity to neg inputs.
    #     Pinputs = piecewiseLinearPseudoSigmoid_fn (Pinputs, mP.cP, pSlope)

    #     dP = dt*( -oldP*mP.tauP + Pinputs )
    #     # Wiener noise:
    #     dWP = sqrt(dt)*wPsig.*meanSpontP.*randn(size(dP))
    #     # combine them:
    #     newP = oldP + dP + dWP

#-------------------------------------------------------------------------------

    #     # dPI:                                 # no PIs for mnist
    #     PIinputs = -mP.L2PI*oldL.*max( 0, (1 - thisOctoHit*mP.octo2PI*mP.octoNegDiscount) ) + (mP.R2PI*oldR).*(1 + thisOctoHit*mP.octo2PI)

    #     PIinputs = piecewiseLinearPseudoSigmoid_fn (PIinputs, mP.cPI, piSlope)

    #     dPI = dt*( -oldPI*mP.tauPI + PIinputs )
    #     # Wiener noise:
    #     dWPI = sqrt(dt)*wPIsig.*meanSpontPI.*randn(size(dPI))
    #     # combine them:
    #     newPI = oldPI + dPI + dWPI

#-------------------------------------------------------------------------------

    #     # dL:
    #     Linputs = -mP.L2L*oldL.*max( 0, (1 - thisOctoHit*mP.octo2L*mP.octoNegDiscount ) )...
    #         + (mP.R2L.*oldR).*(1 + thisOctoHit*mP.octo2L )


    #     Linputs = piecewiseLinearPseudoSigmoid_fn (Linputs, mP.cL, lSlope)

    #     dL = dt*( -oldL*mP.tauL + Linputs )
    #     # Wiener noise:
    #     dWL = sqrt(dt)*wLsig.*meanSpontL.*randn(size(dL))
    #     # combine them:
    #     newL = oldL + dL + dWL

#-------------------------------------------------------------------------------

    ## Enforce sparsity on the KCs:
    #     # Global damping on KCs is controlled by mP.sparsityTarget (during
    #     # octopamine, by octSparsityTarget). Assume that inputs to KCs form a
    #     # gaussian, and use a threshold calculated via std devs to enforce the correct sparsity.

    #     # Delays from AL -> MB and AL -> LH -> MB (~30 mSec) are ignored.

    #     numNoOctoStds = sqrt(2)*erfinv(1 - 2*mP.sparsityTarget) # the # st devs to give the correct sparsity
    #     numOctoStds = sqrt(2)*erfinv(1 - 2*mP.octoSparsityTarget)
    #     numStds = (1-thisOctoHit)*numNoOctoStds + thisOctoHit*numOctoStds # selects for either octo or no-octo
    #     minDamperVal = 1.2*maxSpontP2KtimesPval # a minimum damping based on spontaneous PN activity, so that the MB is silent absent odor
    #     thisKinput = oldP2K*oldP - oldPI2K*oldPI # (no PIs for mnist, only Ps)
    #     damper = unique( mean(thisKinput) + numStds*std(thisKinput) )
    #     damper = max(damper, minDamperVal)

    #     Kinputs = oldP2K*oldP.*(1 + mP.octo2K*thisOctoHit) ...    # but note that mP.octo2K == 0
    #       - ( damper*kGlobalDampVec + oldPI2K*oldPI ).*max( 0, (1 - mP.octo2K*thisOctoHit) ) # but no PIs for mnist

    #     Kinputs = piecewiseLinearPseudoSigmoid_fn (Kinputs, mP.cK, kSlope)

    #     dK = dt*( -oldK*mP.tauK + Kinputs )
    #     # Wiener noise:
    #     dWK = sqrt(dt)*wKsig.*meanSpontK.*randn(size(dK))
    #     # combine them:
    #     newK = oldK + dK + dWK

#-------------------------------------------------------------------------------

    #     # readout neurons E (EN = 'extrinsic neurons'):
    #     # These are readouts, so there is no sigmoid.
    #     # mP.octo2E == 0, since we are not stimulating ENs with octo.
    #     # dWE == 0 since we assume no noise in ENs.

    #     Einputs = oldK2E*oldK # (oldK2E*oldK).*(1 + thisOctoHit*mP.octo2E) # mP.octo2E == 0

    #     dE = dt*( -oldE*mP.tauE + Einputs )
    #     # Wiener noise:
    #     dWE = 0 #  sqrt(dt)*wEsig.*meanSpontE.*randn(size(dE)) # noise = 0 => dWE == 0
    #     # combine them:
    #     newE = oldE + dE + dWE # always non-neg

#-------------------------------------------------------------------------------

    ## HEBBIAN UPDATES:

    #     # Apply Hebbian learning to mP.P2K, mP.K2E:
    #     # For ease, use 'newK' and 'oldP', 'newE' and 'oldK', ie 1 timestep of delay.
    #     # We restrict hebbian growth in mP.K2E to connections into the EN of the training stimulus

    #     if hebRegion(i)   # Hebbian updates are active for about half the duration of each stimulus

    #         # the PN contribution to hebbian is based on raw FR:
    #         tempP = oldP
    #         tempPI = oldPI # no PIs for mnist
    #         nonNegNewK = max(0,newK) # since newK has not yet been made non-neg

    ## dP2K:
    #         dp2k = (1/mP.hebTauPK) *nonNegNewK * (tempP')
    #         dp2k = dp2k.*P2Kmask # if original synapse does not exist, it will never grow.

    #         # decay some mP.P2K connections if wished: (not used for mnist experiments)
    #         if mP.dieBackTauPK > 0
    #             oldP2K = oldP2K - oldP2K*(1/mP.dieBackTauPK)*dt
    #         end

    #         newP2K = oldP2K + dp2k
    #         newP2K = max(0, newP2K)
    #         newP2K = min(newP2K, mP.hebMaxPK*ones(size(newP2K)))

#-------------------------------------------------------------------------------

    ## dPI2K: # no PIs for mnist
    #         dpi2k = (1/mP.hebTauPIK) *nonNegNewK *(tempPI')
    #         dpi2k = dpi2k.*PI2Kmask # if original synapse does not exist, it will never grow.
    #         # kill small increases:
    #         temp = oldPI2K # this detour prevents dividing by zero
    #         temp(temp == 0) = 1
    #         keepMask = dpi2k./temp
    #         keepMask = reshape(keepMask, size(dpi2k))
    #         dpi2k = dpi2k.*keepMask
    #         if mP.dieBackTauPIK > 0
    #             oldPI2K = oldPI2K - oldPI2K*(1/mP.dieBackTauPIK)*dt
    #         end
    #         newPI2K = oldPI2K + dpi2k
    #         newPI2K = max(0, newPI2K)
    #         newPI2K = min(newPI2K, mP.hebMaxPIK*ones(size(newPI2K)))

#-------------------------------------------------------------------------------

    ## dK2E:
    #         tempK = oldK
    #         dk2e = (1/mP.hebTauKE) * newE* (tempK')  # oldK is already nonNeg
    #         dk2e = dk2e.*K2Emask

    #         # restrict changes to just the i'th row of mP.K2E, where i = ind of training stim
    #         restrictK2Emask = zeros(size(mP.K2E))
    #         restrictK2Emask(thisStimClassInd,:) = 1
    #         dk2e = dk2e.*restrictK2Emask

#-------------------------------------------------------------------------------

    # inactive connections for this EN die back:
    #         if mP.dieBackTauKE > 0
    #             # restrict dieBacks to only the trained EN:
    #             targetMask = zeros(size(dk2e(:)))
    #             targetMask( dk2e(:) == 0 ) = 1
    #             targetMask = reshape(targetMask, size(dk2e))
    #             targetMask = targetMask.*restrictK2Emask
    #             oldK2E = oldK2E - targetMask.*(oldK2E + 2)*(1/mP.dieBackTauKE)*dt # the '+1' allows weights to die to absolute 0
    #         end

    #         newK2E = oldK2E + dk2e
    #         newK2E = max(0,newK2E)
    #         newK2E = min(newK2E, mP.hebMaxKE*ones(size(newK2E)))

    #     else                       # case: no heb or no octo
    #         newP2K = oldP2K
    #         newPI2K = oldPI2K # no PIs for mnist
    #         newK2E = oldK2E
    #     end

#-------------------------------------------------------------------------------

    # update the evolution matrices, disallowing negative FRs.
    #     if T(i) < stopSpontMean3 + 5 || mP.saveAllNeuralTimecourses
    #         R(:,i+1) = max( 0, newR)
    #         P(:,i+1) = max( 0, newP)
    #         PI(:,i+1) = max( 0, newPI) # no PIs for mnist
    #         L(:,i+1) = max( 0, newL)
    #         K(:,i+1) = max( 0, newK)
    #         E(:,i+1) = newE
    #     # case: do not save AL and MB neural timecourses after the noise calibration is done, to save on memory
    #     else
    #         R = max( 0, newR)
    #         P = max( 0, newP)
    #         PI = max( 0, newPI) # no PIs for mnist
    #         L  = max( 0, newL)
    #         K  = max( 0, newK)
    #     end

    #     E(:,i+1) = newE # always save full EN timecourses

    # end # for i = 1:N
    # # Time-step simulation is now over.

    # # combine so that each row of fn output Y is a col of [P; PI; L; R; K]:
    # if mP.saveAllNeuralTimecourses
    #     Y = vertcat(P, PI, L, R, K, E)
    #     Y = Y'
    #     thisRun.Y = single(Y) # convert to singles to save memory
    # else
    #     thisRun.Y = []
    # end

    # thisRun.T = single(T') # store T as a col
    # thisRun.E = single(E') # length(T) x mP.nE matrix
    # thisRun.P2Kfinal = single(oldP2K)
    # thisRun.K2Efinal = single(oldK2E)
    # end


    return thisRun
