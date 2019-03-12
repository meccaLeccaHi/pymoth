def initializeConnectionMatrices(mP):
    # Generates the various connection matrices, given a modelParams object,
    # and appends them to modelParams.
    # Input: 'mP', model params object
    # Output: 'params', a struct that includes connection matrices and other model
    # info necessary to FR evolution and plotting

    #--------------------------------------------------------------------

    # step 1: build the matrices
    # step 2: pack the matrices into a struct 'params' for output
    # These steps are kept separate for clarity of step 2

    #--------------------------------------------------------------------

    ## Step 1: Generate connection matrices
    # Comment: Since there are many zero connections (ie matrices are usually
    # not all-to-all) we often need to apply masks to preserve the zero connections

    # l = dir(mP)
    # d = mP.__dict__
    # from pprint import pprint
    # pprint(l)
    # pprint(d, indent=2)

    import numpy as np
    import numpy.random as r
    from decimal import Decimal, getcontext
    getcontext().prec = 4 # set Decimal type precision

    def pos_rect(input):
        '''Positive rectifier function'''
        input[input < 0] = 0
        return input

    # first make a binary mask S2Rbinary
    if mP.RperFFrMu > 0:
        mP.F2Rbinary = r.rand(mP.nR, mP.nF) < mP.RperSFrMu # 1s and 0s
        if mP.makeFeaturesOrthogonalFlag:
            # remove any overlap in the active odors, by keeping only one non-zero entry in each row
            b = mP.F2Rbinary
            for i in range(mP.nR):
                row = b[i,:]
                if row.sum() > 1:
                    c = np.where(row==1)
                    t = np.ceil(r.rand(1, c)) # pick one index to be non-zero
                    b[i,:] = 0
                    b[i,c[t]] = 1
            mP.F2Rbinary = b

    else: # case: we are assigning a fixed # gloms to each S
        mP.F2Rbinary = np.zeros((mP.nR, mP.nF))
        counts = np.zeros((mP.nR,1)) # to track how many S are hitting each R
        # calc max n of S per any given glom
        maxFperR = np.ceil(mP.nF*mP.RperFRawNum/mP.nR)
        # connect one R to each S, then go through again to connect a 2nd R to each S, etc
        for i in range(mP.RperFRawNum):
            for j in range(mP.nF):
                inds = np.where(counts < maxFperR)
                a = np.random.randint(len(inds))
                counts[inds[a]] += 1
                mP.F2Rbinary[inds[a],j] = 1

    # now mask a matrix of gaussian weights
    rand_mat = r.rand(*mP.F2Rbinary.shape)
    mP.F2R = ( mP.F2Rmu*mP.F2Rbinary + mP.F2Rstd*rand_mat )*mP.F2Rbinary # the last term ensures 0s stay 0s
    mP.F2R = pos_rect(mP.F2R) # to prevent any negative weights

    # spontaneous FRs for Rs
    if mP.spontRdistFlag==1: # case: gaussian distribution
        mP.Rspont = mP.spontRmu*np.ones((mP.nG, 1)) + mP.spontRstd*r.rand(mP.nG, 1)
        mP.Rspont = pos_rect(mP.Rspont)
    else: # case: 2 gamma distribution
        a = Decimal(mP.spontRmu)/Decimal(mP.spontRstd)
        b = Decimal(mP.spontRmu)/a # spontRstd
        g = np.random.gamma(a, scale=b, size=(mP.nG,1))
        mP.Rspont = mP.spontRbase + g

    # R2G connection vector: nG x 1 col vector
    mP.R2G = mP.R2Gmu*np.ones((mP.nG, 1)) + mP.R2Gstd*r.rand(mP.nG, 1) # col vector,
    # each entry is strength of an R in its G. the last term prevents negative R2G effects

    # now make R2P, etc, all are cols nG x 1
    mP.R2P = ( mP.R2Pmult + mP.R2Pstd*r.rand(mP.nG, 1) )*mP.R2G
    mP.R2L = ( mP.R2Lmult + mP.R2Lstd*r.rand(mP.nG, 1) )*mP.R2G

    # this interim nG x 1 col vector gives the effect of each R on any PI in the R's glom.
    mP.R2PIcol = ( mP.R2PImult + mP.R2PIstd*r.rand(mP.nG, 1) )*mP.R2G
    # It will be used below with G2PI to get full effect of Rs on PIs

    # Construct L2G = nG x nG matrix of lateral neurons. This is a precursor to L2P etc
    # DEV NOTE: Had to make this different than matlab version - in Python, scalars have no shape
    # Run by CBD
    mP.L2G = mP.L2Gmu + mP.L2Gstd*r.rand(mP.nG, mP.nG)
    mP.L2G = pos_rect(mP.L2G) # kill any vals < 0
    mP.L2G -= np.diag(np.diag(mP.L2G)) # set diagonal = 0

    # are enough of these values 0?
    numZero = (mP.L2G.flatten()==0).sum() - mP.nG # ignore the diagonal zeroes
    numToKill = np.floor( (1-mP.L2Gfr)*(mP.nG**2 - mP.nG) - numZero )
    if numToKill > 0: # case: we need to set more vals to 0 to satisfy frLN constraint
        mP.L2G = mP.L2G.flatten()
        randList = r.rand(*mP.L2G.shape) < numToKill/(mP.nG**2 - mP.nG - numZero)
        mP.L2G[(mP.L2G > 0) & (randList == 1)] = 0

    mP.L2G = mP.L2G.reshape((mP.nG,mP.nG), order="F") # DEV NOTE: Using Fortran order (as MATLAB does)
    # Structure of L2G:
    # L2G(i,j) = the synaptic LN weight going to G(i) from G(j),
    # ie the row gives the 'destination glom', the col gives the 'source glom'

    # gloms vary widely in their sensitivity to gaba (Hong, Wilson 2014).
    # multiply the L2* vectors by Gsens + GsensStd:
    gabaSens = mP.GsensMu + mP.GsensStd*r.rand(mP.nG, 1)
    L2GgabaSens = mP.L2G * np.tile( gabaSens, (1, mP.nG) ) # ie each row is multiplied by a different value,
        # since each row represents a destination glom
    # this version of L2G does not encode variable sens to gaba, but is scaled by GsensMu:
    mP.L2G *= mP.GsensMu

    # now generate all the L2etc matrices:
    mP.L2R = pos_rect( mP.L2Rmult + mP.L2Rstd*r.rand(mP.nG,mP.nG) ) * L2GgabaSens
     # the last term will keep 0 entries = 0
    mP.L2P = pos_rect( mP.L2Pmult + mP.L2Pstd*r.rand(mP.nG,mP.nG) ) * L2GgabaSens
    mP.L2L = pos_rect( mP.L2Lmult + mP.L2Lstd*r.rand(mP.nG,mP.nG) ) * L2GgabaSens
    mP.L2PI = pos_rect( mP.L2Lmult + mP.L2PIstd*r.rand(mP.nG,mP.nG) ) * L2GgabaSens
     # Masked by G2PI later

    # Ps (excitatory):
    P2KconnMatrix = r.rand(mP.nK, mP.nP) < mP.KperPfrMu # each col is a P, and a fraction of the entries will = 1
     # different cols (PNs) will have different numbers of 1's (~binomial dist)

    mP.P2K = pos_rect( mP.P2Kmu + mP.P2Kstd*r.rand(mP.nK, mP.nP) ) # all >= 0
    mP.P2K *= P2KconnMatrix
    # cap P2K values at hebMaxP2K, so that hebbian training never decreases wts:
    mP.P2K[mP.P2K < mP.hebMaxPK] = mP.hebMaxPK
    # PKwt maps from the Ps to the Ks. Given firing rates P, PKwt gives the
    # effect on the various Ks
    # It is nK x nP with entries >= 0.


    #--------------------------------------------------------------------
    # PIs (inhibitory): (not used in mnist)
    # 0. These are more complicated, since each PI is fed by several Gs
    # 1. a) We map from Gs to PIs (binary, one G can feed multiple PI) with G2PIconn
    # 1. b) We give wts to the G-> PI connections. these will be used to calc PI firing rates.
    # 2. a) We map from PIs to Ks (binary), then
    # 2. b) Multiply the binary map by a random matrix to get the synapse weights.

    # In the moth, each PI is fed by many gloms
    mP.G2PIconn = r.rand(mP.nPI, mP.nG) < mP.GperPIfrMu # step 1a
    mP.G2PI = pos_rect(mP.G2PIstd*r.rand(mP.nPI, mP.nG) + mP.G2PImu) # step 1b
    mP.G2PI *= mP.G2PIconn # mask with double values, step 1b (cont)
    mP.G2PI /= np.tile(mP.G2PI.sum(axis=1).reshape(-1, 1),(1, mP.G2PI.shape[1]))

    # mask PI matrices
    mP.L2PI = np.matmul(mP.G2PI,mP.L2G) # nPI x nG

    mP.R2PI = mP.G2PI*mP.R2PIcol.T
    # nG x nPI matrices, (i,j)th entry = effect from j'th object to i'th object.
    # eg, the rows with non-zero entries in the j'th col of L2PI are those PIs affected by the LN from the j'th G.
    # eg, the cols with non-zero entries in the i'th row of R2PI are those Rs feeding gloms that feed the i'th PI.

    if mP.nPI>0:
        mP.PI2Kconn = r.rand(mP.nK, mP.nPI) < mP.KperPIfrMu # step 2a
        mP.PI2K = pos_rect(mP.PI2Kmu + mP.PI2Kstd*r.rand(mP.nK, mP.nPI)) # step 2b
        mP.PI2K *= mP.PI2Kconn # mask
        mP.PI2K[mP.PI2K < mP.hebMaxPIK] = mP.hebMaxPIK
        # 1. G2PI maps the Gs to the PIs. It is nPI x nG, doubles.
        #    The weights are used to find the net PI firing rate
        # 2. PI2K maps the PIs to the Ks. It is nK x nPI with entries >= 0.
        #    G2K = PI2K*G2PI # binary map from G to K via PIs. not used

    #--------------------------------------------------------------------

    # K2E (excit):
    mP.K2EconnMatrix = r.rand(mP.nE, mP.nK) < mP.KperEfrMu # each col is a K, and a fraction of the entries will = 1.
    #    different cols (KCs) will have different numbers of 1's (~binomial dist).

    mP.K2E = pos_rect( mP.K2Emu + mP.K2Estd*r.rand(mP.nE, mP.nK) ) # all >= 0
    mP.K2E = np.multiply(mP.K2E, mP.K2EconnMatrix)
    mP.K2E[mP.K2E < mP.hebMaxKE] = mP.hebMaxKE
    # K2E maps from the KCs to the ENs. Given firing rates KC, K2E gives the effect on the various ENs.
    # It is nE x nK with entries >= 0.

    # octopamine to Gs and to Ks
    mP.octo2G = pos_rect( mP.octo2Gmu + mP.octo2Gstd*r.rand(mP.nG, 1) ) # intermediate step
    # uniform distribution (experiment)
    mP.octo2G = pos_rect( mP.octo2Gmu + 4*mP.octo2Gstd*r.rand(mP.nG, 1) - 2*mP.octo2Gstd ) # 2*(linspace(0,1,nG) )' );
    mP.octo2K = pos_rect( mP.octo2Kmu + mP.octo2Kstd*r.rand(mP.nK, 1) )
    # each of these is a col vector with entries >= 0

    mP.octo2P = pos_rect( mP.octo2Pmult*mP.octo2G + mP.octo2Pstd*r.rand(mP.nG, 1) ) # effect of octo on P, includes gaussian variation from P to P
    mP.octo2L = pos_rect( mP.octo2Lmult*mP.octo2G + mP.octo2Lstd*r.rand(mP.nG, 1) )
    mP.octo2R = pos_rect( mP.octo2Rmult*mP.octo2G + mP.octo2Rstd*r.rand(mP.nG, 1) )
    #  uniform distributions (experiments)
    mP.octo2P = pos_rect( mP.octo2Pmult*mP.octo2G + 4*mP.octo2Pstd*r.rand(mP.nG, 1) - 2*mP.octo2Pstd )
    mP.octo2L = pos_rect( mP.octo2Lmult*mP.octo2G + 4*mP.octo2Lstd*r.rand(mP.nG, 1) - 2*mP.octo2Lstd )
    mP.octo2R = pos_rect( mP.octo2Rmult*mP.octo2G + 4*mP.octo2Rstd*r.rand(mP.nG, 1) - 2*mP.octo2Rstd )
    # mask and weight octo2PI
    mP.octo2PIwts = mP.G2PI*( mP.octo2PImult*mP.octo2G.T ) # does not include a PI-varying std term
    # normalize this by taking average
    mP.octo2PI = mP.octo2PIwts.sum(axis=1)/mP.G2PIconn.sum(axis=1) # net, averaged effect of octo on PI. Includes varying effects of octo on Gs & varying contributions of Gs to PIs.

    mP.octo2E = pos_rect( mP.octo2Emu + mP.octo2Estd*r.rand(mP.nE, 1) )




    # each neuron has slightly different noise levels for sde use. Define noise vectors for each type:
    # Gaussian versions:
    # mP.noiseRvec = pos_rect( mP.mP.epsRstd + mP.RnoiseSig*r.rand(mP.nR, 1) ) # remove negative noise entries
    # mP.noisePvec = pos_rect( mP.epsPstd + mP.PnoiseSig*r.rand(mP.nP, 1) )
    # mP.noiseLvec = pos_rect( mP.epsLstd + mP.LnoiseSig*r.rand(mP.nG, 1) )
    mP.noisePIvec = pos_rect( mP.noisePI + mP.PInoiseStd*r.rand(mP.nPI, 1) )
    mP.noiseKvec = pos_rect( mP.noiseK + mP.KnoiseStd*r.rand(mP.nK, 1) )
    mP.noiseEvec = pos_rect( mP.noiseE + mP.EnoiseStd*r.rand(mP.nE, 1) )
    # gamma versions:
    a = Decimal(mP.noiseR)/Decimal(mP.RnoiseStd)
    b = Decimal(mP.noiseR)/a
    mP.noiseRvec = np.random.gamma(a, scale=b, size=(mP.nR,1))

    # DEV NOTE: Run below by CBD - Still necessary?
    mP.noiseRvec[mP.noiseRvec > 15] = 0 # experiment to see if just outlier noise vals boost KC noise

    a = Decimal(mP.noiseL)/Decimal(mP.LnoiseStd)
    b = Decimal(mP.noiseL)/a
    mP.noiseLvec = np.random.gamma(a, scale=b, size=(mP.nG,1))

    mP.kGlobalDampVec = mP.kGlobalDampFactor + mP.kGlobalDampStd*r.rand(mP.nK,1)
    # each KC may be affected a bit differently by LH inhibition

    #--------------------------------------------------------------------

    # append these matrices to 'modelParams' struct:
    # no editing necessary in this section

    # modelParams.F2R = F2R;
    # modelParams.R2P = R2P;
    # modelParams.R2PI = R2PI;
    # modelParams.R2L = R2L;
    # modelParams.octo2R = octo2R;
    # modelParams.octo2P = octo2P;
    # modelParams.octo2PI = octo2PI;
    # modelParams.octo2L = octo2L;
    # modelParams.octo2K = octo2K;
    # modelParams.octo2E = octo2E;
    # modelParams.L2P = L2P;
    # modelParams.L2L = L2L;
    # modelParams.L2PI = L2PI;
    # modelParams.L2R = L2R;
    # modelParams.G2PI = G2PI;
    # modelParams.P2K = P2K;
    # modelParams.PI2K = PI2K;
    # modelParams.K2E = K2E;
    # modelParams.Rspont = Rspont;  % col vector

    # modelParams.noiseRvec = noiseRvec;
    # modelParams.noisePvec = noisePvec;
    # modelParams.noisePIvec = noisePIvec;
    # modelParams.noiseLvec = noiseLvec;
    # modelParams.noiseKvec = noiseKvec;
    # modelParams.noiseEvec = noiseEvec;
    # modelParams.kGlobalDampVec = kGlobalDampVec;
