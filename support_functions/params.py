#!/usr/bin/env python3

# import packages
import numpy as np

class ModelParams:
    '''
    This Python module contains the parameters for a sample moth, ie the template
    that is used to populate connection matrices and to control behavior.

    Parameters:
        1. nF = number of features. This determines the number of neurons in each layer.
        2. goal = measure of learning rate: goal = N means we expect the moth to
            hit max accuracy when trained on N samples per class. So goal = 1 gives
            a fast learner, goal = 20 gives a slower learner.

    #-------------------------------------------------------------------------------

    The following abbreviations are used:
    n* = number of *
    G = glomerulus (so eg nG = number of glomeruli)
    R = response neuron (from antennae): this concept is not used.
        We use the stim -> glomeruli connections directly
    P = excitatory projection neuron. note sometimes P's stand in for gloms
        in indexing, since they are 1 to 1
    PI = inhibitory projection neuron
    L = lateral neuron (inhibitory)
    K = kenyon cell (in MB)
    F = feature (this is a change from the original moth/odor regime, where each
    stim/odor was identified with its own single feature)
    S (not used in this function) = stimulus class
    fr = fraction%
    mu = mean
    std = standard deviation
    _2_ = synapse connection to, eg P2Kmu = mean synapse strength from PN to KC
    octo = octopamine delivery neuron

    General structure of synaptic strength matrices:
    rows give the 'from' a synapse
    cols give the 'to'
    so M(i,j) is the strength from obj(i) to obj(j)

    below, 'G' stands for glomerulus. Glomeruli are not explicitly part of the equations,
    but the matrices for LN interconnections,
    PNs, and RNs are indexed according to the G

    Copyright (c) 2019 Adam P. Jones (ajones173@gmail.com) and Charles B. Delahunt (delahunt@uw.edu)
    MIT License
    '''
    def __init__(self, nF, goal):

        self.nF = nF
        self.goal = goal

        self.nG = nF
        self.nP = self.nG # Pn = n of excitatory Pn. (one per glomerulus)
        self.nR = self.nG # RNs (one per glom)

        # for now assume no pheromone gloms. Can add later, along with special Ps, PIs, and Ls
        self.nK2nGRatio = 30
        self.nK = int(self.nK2nGRatio)*int(self.nG) # number of kenyon cells (in MB)
        # enforcing integer multiplication above

        # get count of inhibitory projection neurons = PIs
        # for mnist experiments there are no inhibitory projection neurons (PIs)
        self.PIfr = 0.05  # But make a couple placeholders
        self.nPI = int(self.nG*self.PIfr) # these are in addition to nP
        # note that outputs P and PI only affect KCs
        self.nE = 10 # extrinsic neurons in eg beta-lobe of MB,
        # ie the read-out/decision neurons

        #-------------------------------------------------------------------------------

        ## Hebbian learning rates:

        # For K2E ie MB -> EN. Very important. Most of the de facto plasticity is in K2E:
        self.hebTauKE = 0.02*goal # controls learning rate for K2E weights. 1/decay rate.
        # Higher means slower decay.

        self.dieBackTauKE = 0.5*goal # 1/decay rate. Higher means slower decay.
        # dieBackTauKE and hebTauKE want to be in balance.

        # For P2K ie AL -> MB
        self.hebTauPK = 5e3*goal # learning rate for P2K weights. Higher means slower.
        # Very high hebTauPK means that P2K connections are essentially fixed.
        # Decay: There is no decay for P2K weights
        self.dieBackTauPK = 0 # If > 0, divide this fraction of gains evenly among all nonzero
        # weights, and subtract.
        self.dieBackTauPIK = 0 # no PIs in mnist moths (no PIs for mnist)

        #-------------------------------------------------------------------------------

        ## Time constants for the diff eqns

        # An important param: the same param set will give different results if only
        # this one is changed.
        # Assume: 3*(1/time constant) = approx 1+ seconds (3 t.c -> 95% decay)
        self.tau = 7
        self.tauR = self.tau
        self.tauP = self.tau
        self.tauPI = self.tau # no PIs for mnist
        self.tauL = self.tau
        self.tauK = self.tau
        self.tauE = self.tau

        # stdmoid range parameter for the diff eqns
        self.C = 10.5
        self.cR = self.C
        self.cP = self.C
        self.cPI = self.C # no PIs for mnist
        self.cL = self.C
        self.cK = self.C

        # stdmoid slope param for the diff eqns
        # slope of stdmoid at zero = C*slopeParam/4
        self.desiredSlope = 1 # 'desiredSlope' is the one to adjust.
        self.slopeParam = self.desiredSlope*4/self.C # a convenience variable.
        # slope of sigmoid at 0 = slopeParam*c/4, where c = self.cR, self.cP, self.cL, etc

        # stdmoid param to make range 0:C or -C/2:C/2
        self.symmetricAboutZero = 1 # '0' means: [-inf:inf] -> [0, C], 0 -> C/2.
        # '1' means: [-inf:inf] -> [-c/2,c/2], 0 -> 0.
        self.stdFactor= 0.1 # this value multiplies the mean value of a connection matrix
        # to give the STD. It applies to all connection matrices, ie it is a global parameter.

        ## Parameters to generate connection matrices between various types of neurons:

        # Typically the effect of an input to G (ie a glomerulus), as passed on to P,
        # L, PI, and R within the glomerulus, is set = mult*effectOnG + std*NormalDist.
        # 'mult' is different for P, L, and R. That is, a stdnal entering the glomerulus
        # affects the different neuron types differently.

        # mu and std define mean and std of connection matrix entries

        # KEY POINT: R2L, R2P, and R2PI should be highly correlated, since all of them
        # depend on the R2G connectivity strength. For this reason, use an
        # intermediate connectivity matrix R2G to give the base (glomerular) levels of
        # connectivity. Then derive R2P, R2L from R2G. Similarly, define L2G
        # first, then derive L2L, L2P, L2R).

        # For mnist, each pixel is a feature, and goes to exactly one receptor
        # neuron RN, ie to exactly one glomerulus G
        self.RperFFrMu = 0 # used in odor simulations, disabled for mnist.
        self.RperFRawNum = 1 # ie one RN (equiv one Glom) for each F (feature, ie pixel).

        self.F2Rmu = 100/self.nG # controls how strongly a given feature will affect G's
        self.F2Rstd = 0

        #-------------------------------------------------------------------------------

        # R characteristics
        self.R2Gmu = 5 # controls how strongly R's affect their G's
        self.R2Gstd = 0

        # used to create R2P, using as base R2G
        self.R2Pmult = 4 # this multiplies R2G values, to give R2P values
        self.R2Pstd = self.stdFactor*self.R2Pmult # variation in R2P values, beyond the conversion from R2G values

        self.R2PImult = 0 # no PIs for mnist
        self.R2PIstd = 0

        self.R2Lmult = 0.75 # so R has much weaker effect on LNs than on PNs (0.75 vs 4)
        self.R2Lstd = self.stdFactor*self.R2Lmult

        #-------------------------------------------------------------------------------

        # define spontaneous steady-state firing rates (FRs) of RNs
        self.spontRdistFlag = 2 # 1 = gaussian, 2 = gamma + base.
        self.spontRmu = 0.1
        self.spontRstd = 0.05*self.spontRmu
        self.spontRbase = 0 # for gamma only
        # if using gamma, params are derived from spontRmu and spontRstd.

        #-------------------------------------------------------------------------------

        # L2* connection matrices

        # KEY POINT: L2R, L2L, L2P, and L2PI should be highly correlated, since all
        # depend on the L2G connectivity strength. For this reason, use an
        # intermediate connectivity matrix L2G to give the base levels of
        # connectivity. Then derive L2P, L2R, L2L from L2G.

        # Define the distribution of LNs, ie inhib glom-glom synaptic strengths:
        self.L2Gfr = 0.8 # the fraction of possible LN connections (glom to glom) that are non-zero.
        # hong: fr = most or all
        self.L2Gmu = 5
        self.L2Gstd = self.stdFactor*self.L2Gmu

        # sensitivity of G to gaba, ie sens to L, varies substantially with G.
        # Hong-Wilson says PNs (also Gs) sensitivity to gaba varies as N(0.4,0.2).
        # Note this distribution is the end-result of many interactions, not a direct
        # application of GsensMu and Gsensstd.
        self.GsensMu = 0.6
        self.GsensStd = 0.2*self.GsensMu
        # Note: Gsensstd expresses variation in L effect, so if we assume that P, L,
        # and R are all similarly gaba-resistent within a given G, set L2Pstd etc below = 0.

        #-------------------------------------------------------------------------------

        ## define effect of LNs on various components

        # used to create L2R, using as base L2G
        self.L2Rmult = 6/ self.nG*self.L2Gfr
        self.L2Rstd = 0.1*self.L2Rmult
        # used to create L2P, using as base L2G
        self.L2Pmult = 2/ self.nG*self.L2Gfr
        self.L2Pstd = self.stdFactor*self.L2Pmult # variation in L2P values, beyond the conversion from L2G values
        # used to create L2PI, using as base L2G
        self.L2PImult = self.L2Pmult
        self.L2PIstd = self.stdFactor*self.L2PImult
        # used to create L2L, using as base L2G
        self.L2Lmult = 2/ self.nG*self.L2Gfr
        self.L2Lstd = self.stdFactor*self.L2Lmult

        # weights of G's contributions to PIs, so different Gs will contribute differently to net PI FR:
        self.G2PImu = 1 # 0.6
        self.G2PIstd = 0 # 0.1*G2PImu

        #-------------------------------------------------------------------------------

        # Sparsity in the KCs:
        # KCs are globally damped by the LH (or perhaps by a neuron within the MB). The
        # key net effect is sparsity in the KCs.
        # For mnist experiments, the KC sparsity level is directly controlled by two parameters.

        self.sparsityTarget = 0.05
        self.octoSparsityTarget = 0.075 # used when octopamine is active

        self.kGlobalDampFactor = 1 # used to make a vector to deliver damping to KCs
        self.kGlobalDampStd = 0 # allows variation of damping effect by KC.
        # Effect of the above is to make global damping uniform on KCs.

        #-------------------------------------------------------------------------------

        # AL to MB, ie PN to KC connection matrices:
        #
        # Define distributions to describe how Ps connect to Ks, ie synapses:
        #
        # a) excitatory PNs. We need # of KCs connected to each PN:
        # first give the # Ps that feed each K
        self.numPperK = 10
        # mean fraction of KCs a given 'PN' connects to
        self.KperPfrMu = self.numPperK / float(self.nG) # forcing float division
            # Note that here 'PN' = glom. That is, multiple PNs coming
            # out of a single glom are treated as one PN.
            # assuming 5 true PNs per glom, and 2000 KCs, 0.2 means:
            # each glom -> 0.2*2000 = 400 KC, so each true PN -> 80
            # KCs, so there are 300*80 PN:KC links = 24000 links, so
            # each KC is linked to 24k/2k = 12 true PNs (see Turner 2008).
            # The actual # will vary as a binomial distribution
            # simpler calc: KperPfrMu*nP = "true" PNs per K.
            # The end result is a relatively sparse P2K connection
            # matrix with very many zeros. The zeros are permanent (not
            # modifiable by plasticity).

        # b) inhibitory PNs (not used in mnist experiments, but use placeholders to avoid crashes. Weights are 0).
        # We need the # of Gs feeding into each PI and # of Ks the PI goes to:
        self.KperPIfrMu = 1 / float(self.nK) # KperPfrMu; the ave fraction of KCs a PI connects to
        # The actual # will vary as a binomial distribution
        self.GperPIfrMu = 5 / float(self.nG)  # ave fraction of Gs feeding into a PI (for Ps, there is always 1 G)
        # The actual # will vary as a binomial distribution
        # real moth: about 3 - 8 which corresponds to GperPIfrMu = 0.1

        # Define distribution that describes the strength of PN->KC synapses:
        # these params apply to both excitatory and inhibitory PNs. These are plastic, via
        # PN stimulation during octo stim.
        self.pMeanEstimate = (self.R2Gmu*self.R2Pmult)*(self.spontRbase + self.spontRmu) # ~2 hopefully. For use in calibrating P2K
        self.piMeanEstimate = self.pMeanEstimate*self.KperPfrMu/self.KperPIfrMu # not used in mnist

        self.P2Kmultiplier = 44

        self.P2Kmu = self.P2Kmultiplier / float(self.numPperK*self.pMeanEstimate)
        self.P2Kstd = 0.5*self.P2Kmu # bigger variance than most connection matrices

        self.PI2Kmu = 0
        self.PI2Kstd = 0

        self.hebMaxPK = self.P2Kmu + (3*self.P2Kstd) # ceiling for P2K connection weights
        self.hebTauPIK = self.hebTauPK  # irrelevant since PI2K weights == 0 (no PIs for mnist)
        self.hebMaxPIK = self.PI2Kmu + (3*self.PI2Kstd) # no PIs for mnist

        #-------------------------------------------------------------------------------

        # KC -> EN connections:
        # Start with all weights uniform and full connectivity
        # Training rapidly individuates the connectivities of ENs
        self.KperEfrMu = 1 # what fraction of KCs attach to a given EN
        self.K2Emu = 3 # strength of connection
        self.K2Estd = 0 # variation in connection strengths
        self.hebMaxKE = 20*self.K2Emu + 3*self.K2Estd # max allowed connection weights (min = 0)

        #-------------------------------------------------------------------------------

        # distribution of octopamine -> glom strengths (small variation):
        # NOTES:
        # 1. these values assume octoMag = 1
        # 2. if using the multiplicative effect, we multiply the inputs to neuron N by (1 +
        #   octo2N) for positive inputs, and (1-octo2N) for negative inputs. So we
        #   probably want octo2N to be close to 0. It is always non-neg by
        #   construction in 'init_connection_matrix.m'

        # In the context of dynamics eqns, octopamine reduces a neuron's responsivity to inhibitory inputs, but
        # it might do this less strongly than it increases the neuron's responsivity to excitatory inputs:
        self.octoNegDiscount = 0.5 # < 1 means less strong effect on neg inputs.

        # since octo strengths are correlated with a glom, use same method as for gaba sensitivity
        self.octo2Gmu = 1.5
        self.octoStdFactor= 0 # ie make octo effect on all gloms the same
        self.octo2Gstd = self.octoStdFactor * self.octo2Gmu

        # Per jeff (may 2016), octo affects R, and may also affect P, PI, and L
        # First try "octo affects R only", then add the others
        # used to create octo2R, using as base octo2G
        self.octo2Rmult = 2
        self.octo2Rstd = self.octoStdFactor * self.octo2Rmult
        # used to create octo2P, using as base octo2G
        self.octo2Pmult = 0 # octo2P = 0 means we only want to stimulate the response to
        # actual odor inputs (ie at R)
        #  octo2Pmult > 0 mean we stimulate P's responsiveness to all signals, so we
        # amplify noise as well
        self.octo2Pstd = self.octoStdFactor * self.octo2Pmult
        # used to create octo2PI, using as base octo2G:  unused for mnist experiments
        self.octo2PImult = 0
        self.octo2PIstd = self.octo2Pstd

        # used to create octo2L, using as base octo2G
        self.octo2Lmult = 1.6 # This must be > 0 if Rspont is not affected by octo in sde_dynamics
        # (ie if octo only affects RN's reaction to odor), since jeff's data shows
        # that some Rspont decrease with octopamine
        self.octo2Lstd = self.octoStdFactor * self.octo2Lmult

        # end of AL-specific octopamine param specs

        # Distribution of octopamine -> KC strengths (small variation):
        # We assume no octopamine stimulation of KCs, ie higher KC activity follows
        # from higher AL activity due to octopamine, not from direct octopamine effects on KCs.
        self.octo2Kmu = 0
        self.octo2Kstd = self.octoStdFactor * self.octo2Kmu

        self.octo2Emu = 0  # for completeness, not used
        self.octo2Estd = 0

        #-------------------------------------------------------------------------------

        # Noise parameters for noisy AL neurons
        # - Define distributions for random variation in P,R,L and K vectors at each step
        # of the simulation
        # - These control random variations in the various neurons that are applied at
        # each time step as 'epsG' and 'epsK'
        # - This might serve to break up potential oscillations
        self.noise = 1 # set to zero for noise free moth
        self.noiseStdFactor= 0.3
        self.noiseR,self.noiseP,self.noiseL,self.noisePI = [self.noise]*4
        self.RnoiseStd,self.PnoiseStd,self.LnoiseStd,self.PInoiseStd = [self.noise*self.noiseStdFactor]*4
        self.noiseK,self.noiseE,self.EnoiseStd = [0]*3
        self.KnoiseStd = self.noiseK*self.noiseStdFactor

        #-------------------------------------------------------------------------------

        # Pre-allocate connection matrix attributes for later
        self.trueClassLabels = None
        self.saveAllNeuralTimecourses = None

    def init_connection_matrix(self):
        '''
        Generates the various connection matrices, given a modelParams object,
        and appends them to modelParams.
        Parameters: model params object
        Returns: 'params', a struct that includes connection matrices and other model
            info necessary to FR evolution and plotting

        #---------------------------------------------------------------------------

        step 1: build the matrices
        step 2: pack the matrices into a struct 'params' for output
            These steps are kept separate for clarity of step 2

        #---------------------------------------------------------------------------

        Step 1: Generate connection matrices
        Comment: Since there are many zero connections (ie matrices are usually
            not all-to-all) we often need to apply masks to preserve the zero connections

        Copyright (c) 2019 Adam P. Jones (ajones173@gmail.com) and Charles B. Delahunt (delahunt@uw.edu)
        MIT License
        '''

        # import numpy as np
        import numpy.random as r

        # first make a binary mask S2Rbinary
        if self.RperFFrMu > 0:
            self.F2Rbinary = r.rand(self.nR, self.nF) < self.RperSFrMu # 1s and 0s
            # DEV NOTE: The following flag doesn't exist - remove? Check w/ CBD
            if self.makeFeaturesOrthogonalFlag:
                # remove any overlap in the active odors, by keeping only one non-zero entry in each row
                b = self.F2Rbinary
                for i in range(self.nR):
                    row = b[i,:]
                    if row.sum() > 1:
                        c = np.nonzero(row==1)[0]
                        t = np.ceil(r.rand(1, c)) # pick one index to be non-zero
                        b[i,:] = 0
                        b[i,c[t]] = 1
                self.F2Rbinary = b

        else: # case: we are assigning a fixed # gloms to each S
            self.F2Rbinary = np.zeros((self.nR, self.nF))
            counts = np.zeros((self.nR,1)) # to track how many S are hitting each R
            # calc max n of S per any given glom
            maxFperR = np.ceil(self.nF*self.RperFRawNum/self.nR)
            # connect one R to each S, then go through again to connect a 2nd R to each S, etc
            for i in range(self.RperFRawNum):
                for j in range(self.nF):
                    inds = np.nonzero(counts < maxFperR)[0]
                    a = np.random.randint(len(inds))
                    counts[inds[a]] += 1
                    self.F2Rbinary[inds[a],j] = 1

        # now mask a matrix of gaussian weights
        rand_mat = r.normal(0,1,self.F2Rbinary.shape)
        # Note: S (stimuli) for odor case is replaced by F (features) for MNIST version
        self.F2R = ( self.F2Rmu*self.F2Rbinary + self.F2Rstd*rand_mat )*self.F2Rbinary # the last term ensures 0s stay 0s
        self.F2R = np.maximum(0, self.F2R) # to prevent any negative weights

        # spontaneous FRs for Rs
        if self.spontRdistFlag==1: # case: gaussian distribution
            #  steady-state RN FR, base + noise:
            self.Rspont = self.spontRmu*np.ones((self.nG, 1)) + self.spontRstd*r.normal(0,1,(self.nG,1))
            self.Rspont = np.maximum(0, self.Rspont)
        else: # case: 2 gamma distribution
            a = self.spontRmu/self.spontRstd
            b = self.spontRmu/a # spontRstd
            g = np.random.gamma(a, scale=b, size=(self.nG,1))
            self.Rspont = self.spontRbase + g

        # R2G connection vector: nG x 1 col vector
        self.R2G = self.R2Gmu*np.ones((self.nG, 1)) + self.R2Gstd*r.normal(0,1,(self.nG,1)) # col vector,
        # each entry is strength of an R in its G. the last term prevents negative R2G effects

        # now make R2P, etc, all are cols nG x 1
        self.R2P = ( self.R2Pmult + self.R2Pstd*r.normal(0,1,(self.nG,1)) )*self.R2G
        self.R2L = ( self.R2Lmult + self.R2Lstd*r.normal(0,1,(self.nG,1)) )*self.R2G

        # this interim nG x 1 col vector gives the effect of each R on any PI in the R's glom.
        self.R2PIcol = ( self.R2PImult + self.R2PIstd*r.normal(0,1,(self.nG,1)) )*self.R2G
        # It will be used below with G2PI to get full effect of Rs on PIs

        # Construct L2G = nG x nG matrix of lateral neurons. This is a precursor to L2P etc
        self.L2G = self.L2Gmu + self.L2Gstd*r.normal(0,1,(self.nG, self.nG))
        self.L2G = np.maximum(0, self.L2G) # kill any vals < 0
        self.L2G -= np.diag(np.diag(self.L2G)) # set diagonal = 0

        # are enough of these values 0?
        numZero = (self.L2G.flatten()==0).sum() - self.nG # ignore the diagonal zeroes
        numToKill = np.floor( (1-self.L2Gfr)*(self.nG**2 - self.nG) - numZero )
        if numToKill > 0: # case: we need to set more vals to 0 to satisfy frLN constraint
            self.L2G = self.L2G.flatten()
            randList = r.rand(*self.L2G.shape) < numToKill/(self.nG**2 - self.nG - numZero)
            self.L2G[(self.L2G > 0) & (randList == 1)] = 0

        self.L2G = self.L2G.reshape((self.nG,self.nG), order="F") # using Fortran order (as MATLAB does)
        # Structure of L2G:
        # L2G(i,j) = the synaptic LN weight going to G(i) from G(j),
        # ie the row gives the 'destination glom', the col gives the 'source glom'

        # gloms vary widely in their sensitivity to gaba (Hong, Wilson 2014).
        # multiply the L2* vectors by Gsens + GsensStd:
        gabaSens = self.GsensMu + self.GsensStd*r.normal(0,1,(self.nG,1))
        L2GgabaSens = self.L2G * np.tile( gabaSens, (1, self.nG) ) # ie each row is multiplied by a different value,
            # since each row represents a destination glom
        # this version of L2G does not encode variable sens to gaba, but is scaled by GsensMu:
        self.L2G *= self.GsensMu

        # now generate all the L2etc matrices:
        self.L2R = np.maximum(0,  self.L2Rmult + self.L2Rstd*r.normal(0,1,(self.nG,self.nG)) ) * L2GgabaSens
         # the last term will keep 0 entries = 0
        self.L2P = np.maximum(0,  self.L2Pmult + self.L2Pstd*r.normal(0,1,(self.nG,self.nG)) ) * L2GgabaSens
        self.L2L = np.maximum(0,  self.L2Lmult + self.L2Lstd*r.normal(0,1,(self.nG,self.nG)) ) * L2GgabaSens
        self.L2PI = np.maximum(0,  self.L2Lmult + self.L2PIstd*r.normal(0,1,(self.nG,self.nG)) ) * L2GgabaSens
         # Masked by G2PI later (no PIs for mnist)

        # Ps (excitatory):
        P2KconnMatrix = r.rand(self.nK, self.nP) < self.KperPfrMu # each col is a P, and a fraction of the entries will = 1
         # different cols (PNs) will have different numbers of 1's (~binomial dist)

        self.P2K = np.maximum(0,  self.P2Kmu + self.P2Kstd*r.normal(0,1,(self.nK, self.nP)) ) # all >= 0
        self.P2K *= P2KconnMatrix
        # cap P2K values at hebMaxP2K, so that hebbian training never decreases wts:
        self.P2K[self.P2K > self.hebMaxPK] = self.hebMaxPK
        # PKwt maps from the Ps to the Ks. Given firing rates P, PKwt gives the
        # effect on the various Ks
        # It is nK x nP with entries >= 0.

    #-------------------------------------------------------------------------------

        # PIs (inhibitory): (not used in mnist)
        # 0. These are more complicated, since each PI is fed by several Gs
        # 1. a) We map from Gs to PIs (binary, one G can feed multiple PI) with G2PIconn
        # 1. b) We give wts to the G-> PI connections. these will be used to calc PI firing rates.
        # 2. a) We map from PIs to Ks (binary), then
        # 2. b) Multiply the binary map by a random matrix to get the synapse weights.

        # In the moth, each PI is fed by many gloms
        self.G2PIconn = r.rand(self.nPI, self.nG) < self.GperPIfrMu # step 1a
        self.G2PI = np.maximum(0, self.G2PIstd*r.normal(0,1,(self.nPI,self.nG)) + self.G2PImu) # step 1b
        self.G2PI *= self.G2PIconn # mask with double values, step 1b (cont)
        self.G2PI /= np.tile(self.G2PI.sum(axis=1).reshape(-1, 1),(1, self.G2PI.shape[1]))
        # no PIs for mnist

        # mask PI matrices
        self.L2PI = np.matmul(self.G2PI,self.L2G) # nPI x nG

        self.R2PI = self.G2PI*self.R2PIcol.T # no PIs for MNIST
        # nG x nPI matrices, (i,j)th entry = effect from j'th object to i'th object.
        # eg, the rows with non-zero entries in the j'th col of L2PI are those PIs affected by the LN from the j'th G.
        # eg, the cols with non-zero entries in the i'th row of R2PI are those Rs feeding gloms that feed the i'th PI.

        if self.nPI>0:
            self.PI2Kconn = r.rand(self.nK, self.nPI) < self.KperPIfrMu # step 2a
            self.PI2K = np.maximum(0, self.PI2Kmu + self.PI2Kstd*r.normal(0,1,(self.nK,self.nPI))) # step 2b
            self.PI2K *= self.PI2Kconn # mask
            self.PI2K[self.PI2K > self.hebMaxPIK] = self.hebMaxPIK

            # no PIs for mnist
            # 1. G2PI maps the Gs to the PIs. It is nPI x nG, doubles.
            #    The weights are used to find the net PI firing rate
            # 2. PI2K maps the PIs to the Ks. It is nK x nPI with entries >= 0.
            #    G2K = PI2K*G2PI # binary map from G to K via PIs. not used

    #-------------------------------------------------------------------------------

        # K2E (excit):
        self.K2EconnMatrix = r.rand(self.nE, self.nK) < self.KperEfrMu # each col is a K, and a fraction of the entries will = 1.
        #    different cols (KCs) will have different numbers of 1's (~binomial dist).

        self.K2E = np.maximum(0,  self.K2Emu + self.K2Estd*r.normal(0,1,(self.nE,self.nK)) ) # all >= 0
        self.K2E = np.multiply(self.K2E, self.K2EconnMatrix)
        self.K2E[self.K2E > self.hebMaxKE] = self.hebMaxKE
        # K2E maps from the KCs to the ENs. Given firing rates KC, K2E gives the effect on the various ENs.
        # It is nE x nK with entries >= 0.

        # octopamine to Gs and to Ks
        self.octo2G = np.maximum(0,  self.octo2Gmu + self.octo2Gstd*r.normal(0,1,(self.nG,1)) ) # intermediate step
        # uniform distribution (experiment)
        # self.octo2G = np.maximum(0,  self.octo2Gmu + 4*self.octo2Gstd*r.rand(self.nG, 1) - 2*self.octo2Gstd ) # 2*(linspace(0,1,nG) )' )
        self.octo2K = np.maximum(0,  self.octo2Kmu + self.octo2Kstd*r.normal(0,1,(self.nK, 1)) )
        # each of these is a col vector with entries >= 0

        self.octo2P = np.maximum(0,  self.octo2Pmult*self.octo2G + self.octo2Pstd*r.normal(0,1,(self.nG,1)) ) # effect of octo on P, includes gaussian variation from P to P
        self.octo2L = np.maximum(0,  self.octo2Lmult*self.octo2G + self.octo2Lstd*r.normal(0,1,(self.nG,1)) )
        self.octo2R = np.maximum(0,  self.octo2Rmult*self.octo2G + self.octo2Rstd*r.normal(0,1,(self.nG,1)) )
        # #  uniform distributions (experiments)
        # self.octo2P = np.maximum(0,  self.octo2Pmult*self.octo2G + 4*self.octo2Pstd*r.rand(self.nG,1) - 2*self.octo2Pstd )
        # self.octo2L = np.maximum(0,  self.octo2Lmult*self.octo2G + 4*self.octo2Lstd*r.rand(self.nG,1) - 2*self.octo2Lstd )
        # self.octo2R = np.maximum(0,  self.octo2Rmult*self.octo2G + 4*self.octo2Rstd*r.rand(self.nG,1) - 2*self.octo2Rstd )
        # mask and weight octo2PI
        self.octo2PIwts = self.G2PI*( self.octo2PImult*self.octo2G.T ) # does not include a PI-varying std term
        # normalize this by taking average
        self.octo2PI = self.octo2PIwts.sum(axis=1)/self.G2PIconn.sum(axis=1) # net, averaged effect of octo on PI. Includes varying effects of octo on Gs & varying contributions of Gs to PIs.
        # no PIs for mnist

        self.octo2E = np.maximum(0,  self.octo2Emu + self.octo2Estd*r.normal(0,1,(self.nE,1)) )


        # each neuron has slightly different noise levels for sde use. Define noise vectors for each type:
        # Gaussian versions:
        # self.noiseRvec = np.maximum(0,  self.self.epsRstd + self.RnoiseSig*r.normal(0,1,(self.nR,1)) ) # remove negative noise entries
        # self.noisePvec = np.maximum(0,  self.epsPstd + self.PnoiseSig*r.normal(0,1,(self.nP,1)) )
        # self.noiseLvec = np.maximum(0,  self.epsLstd + self.LnoiseSig*r.normal(0,1,(self.nG,1)) )
        self.noisePIvec = np.maximum(0,  self.noisePI + self.PInoiseStd*r.normal(0,1,(self.nPI,1)) ) # no PIs for mnist
        self.noiseKvec = np.maximum(0,  self.noiseK + self.KnoiseStd*r.normal(0,1,(self.nK,1)) )
        self.noiseEvec = np.maximum(0,  self.noiseE + self.EnoiseStd*r.normal(0,1,(self.nE,1)) )

        # gamma versions:
        a = self.noiseR/self.RnoiseStd
        b = self.noiseR/a
        self.noiseRvec = np.random.gamma(a, scale=b, size=(self.nR,1))
        # DEV NOTE: Run below by CBD - Still necessary?
        self.noiseRvec[self.noiseRvec > 15] = 0 # experiment to see if just outlier noise vals boost KC noise

        a = self.noiseP/self.PnoiseStd
        b = self.noiseP/a
        self.noisePvec = np.random.gamma(a, scale=b, size=(self.nR,1))
        # DEV NOTE: Run below by CBD - Still necessary?
        self.noisePvec[self.noisePvec > 15] = 0 # experiment to see if outlier noise vals boost KC noise

        a = self.noiseL/self.LnoiseStd
        b = self.noiseL/a
        self.noiseLvec = np.random.gamma(a, scale=b, size=(self.nG,1))

        self.kGlobalDampVec = self.kGlobalDampFactor + self.kGlobalDampStd*r.normal(0,1,(self.nK,1))
        # each KC may be affected a bit differently by LH inhibition

class ExpParams:
    '''
	This function defines parameters of a time-evolution experiment: overall timing, stim timing and
	strength, octo timing and strength, lowpass window parameter, etc.
	It does book-keeping to allow analysis of the SDE time-stepped evolution of the neural firing rates.
	Parameters:
		1. train_classes: vector of indices giving the classes of the training digits in order.
		The first entry must be nonzero. Unused entries can be filled with -1s if wished.
		2. class_labels: a list of labels, eg 1:10 for mnist
		3. val_per_class: how many digits of each class to use for baseline and post-train

#-------------------------------------------------------------------------------

	Order of time periods:
		1. no event period: allow system to settle to a steady state spontaneous FR baseline
		2. baseline period: deliver a group of digits for each class
		3. no event buffer
		4. training period:  deliver digits + octopamine + allow hebbian updates
		5. no event buffer
		6. post-training period: deliver a group of digits for each class

	Copyright (c) 2019 Adam P. Jones (ajones173@gmail.com) and Charles B. Delahunt (delahunt@uw.edu)
    MIT License
    '''
    def __init__( self, train_classes, class_labels, val_per_class ):

        self.stimMag = 20 # stim magnitudes as passed into AL
        # (See original version in smartAsABug codebase)
        self.stimLength = 0.22
        self.nC = len(class_labels) # the number of classes in this experiment

        ## Define the time span and events:
        self.step = 3 # the time between digits (3 seconds)
        self.trStep = self.step + 2 # allow more time between training digits

        self.sim_start = -30 # use negative start-time for convenience (artifact)

        ## Baseline period:
        # do a loop, to allow gaps between class groups:
        self.baselineTimes = np.empty(0)
        self.startTime = 30
        self.gap = 10
        for i in range(self.nC):
            # vector of timepoints
            self.baselineTimes = np.append(self.baselineTimes,
                range(self.startTime, self.startTime + val_per_class*self.step, self.step) )
            self.startTime = int(np.max(self.baselineTimes) + self.gap)
        # include extra buffer before training
        self.endOfBaseline = int(np.max(self.baselineTimes) + 25)

        ## Training period:
        # vector of timepoints, one digit every 'trStep' seconds
        self.trainTimes = np.array(range(self.endOfBaseline,
            self.endOfBaseline + len(train_classes)*self.trStep, self.trStep))
        # includes buffer before Validation
        self.endOfTrain = int(np.max(self.trainTimes) + 25)

        # Val period:
        # do a loop, to allow gaps between class groups
        self.valTimes = np.empty(0)
        self.startTime = self.endOfTrain
        for i in range(self.nC):
            # vector of timepoints
            self.valTimes = np.append(self.valTimes,
                range(self.startTime, self.startTime + val_per_class*self.step, self.step) )
            self.startTime = int(np.max(self.valTimes) + self.gap)
        self.endOfVal = np.max(self.valTimes) + 4

        ## assemble vectors of stimulus data for export:

        # Assign the classes of each stim. Assign the baseline and val in blocks,
        # and the training stims in the order passed in:

        self.stimStarts = np.hstack(( self.baselineTimes, self.trainTimes, self.valTimes ))

        self.whichClass = np.empty(self.stimStarts.shape) * np.nan
        self.numBaseline = val_per_class*self.nC
        self.numTrain = len(train_classes)
        for c in range(self.nC):
            # the baseline groups
            self.whichClass[ c*val_per_class : (c+1)*val_per_class ] = class_labels[c]

            # the val groups
            self.whichClass[ self.numBaseline + self.numTrain + c*val_per_class : \
                self.numBaseline + self.numTrain + (c+1)*val_per_class ]  = class_labels[c]

        self.whichClass[ self.numBaseline : self.numBaseline + self.numTrain ] = train_classes

        # self.whichClass = whichClass
        # self.stimStarts = stimStarts # starting times
        self.durations = self.stimLength*np.ones( len(self.stimStarts) ) # durations
        self.classMags = self.stimMag*np.ones( len(self.stimStarts) ) # magnitudes

        # octopamine input timing:
        self.octoMag = 1
        self.octoStart = self.trainTimes
        self.durationOcto = 1

        # Hebbian timing: Hebbian updates are enabled 25# of the way into the stimulus, and
        # last until 75% of the way through (ie active during the peak response period)
        self.hebStarts = [i + 0.25*self.stimLength for i in self.trainTimes]
        self.hebDurations = 0.5*self.stimLength*np.ones( len(self.trainTimes) )
        self.startTrain = min(self.hebStarts)
        self.endTrain = max(self.hebStarts) + max(self.hebDurations)

        ## Other time parameters required for time evolution book-keeping:
        # end timepoints for the section used to define mean spontaneous firing rates,
        # in order to calibrate noise.
        # To let the system settle, we recalibrate noise levels to current spontaneous
        # FRs in stages.
        # This ensures that in steady state, noise levels are correct in relation to mean FRs.
        # the numbers 1,2,3 do refer to time periods where spont responses are
        # allowed to settle before recalibration.
        self.startPreNoiseSpontMean1 = -25
        self.stopPreNoiseSpontMean1 = -15
        # Currently no change is made in start/stopSpontMean2.
        # So spontaneous behavior may be stable in this range.
        self.startSpontMean2 = -10
        self.stopSpontMean2 = -5
        # currently, spontaneous behavior is steady-state by startSpontMean3.
        self.startSpontMean3 = 0
        self.stopSpontMean3 = 28

        self.preHebPollTime = min(self.trainTimes) - 5
        self.postHebPollTime = max(self.trainTimes) + 5

        # timePoints for plotting EN responses:
        # spontaneous response periods, before and after training, to view effect of
        # training on spontaneous FRs:
        self.preHebSpontStart = self.startSpontMean3
        self.preHebSpontStop = self.stopSpontMean3
        self.postHebSpontStart = max(self.trainTimes) + 5
        self.postHebSpontStop = min(self.valTimes) - 3

        # hamming filter window parameter (= width of transition zone in seconds).
        # The lp filter is applied to odors and to octo
        self.lpParam =  0.12

        self.sim_stop = max(self.stimStarts) + 10

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
