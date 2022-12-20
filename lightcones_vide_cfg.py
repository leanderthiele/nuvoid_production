#!/usr/bin/env python

from vide.backend.classes import *

continueRun = False

startCatalogStage = 1
endCatalogStage = 3

inputDataDir = '<<<inputDataDir>>>'
workDir = '<<<workDir>>>'

# it's slightly annoying that VIDE doesn't respect multiprocessing...
logDir = '<<<logDir>>>'
figDir = '<<<figDir>>>'

numZobovThreads = <<<numZobovThreads>>>
numZobovDivisions = 2
mergingThreshold = 1e-9

zmin = <<<zmin>>>
zmax = <<<zmax>>>

# potentially need to do some jittering to get VIDE to run successfully
retry_index = <<<retry_index>>>
fakeDensity = 2e-3 * (1 + 0.1*(retry_index/2 if retry_index%2==0 else -(retry_index+1)/2))

dataSampleList = [Sample(dataFile='lightcone_<<<augment>>>.txt',
                         fullName='<<<augment>>>',
                         nickName='<<<augment>>>',
                         dataType='observation',
                         volumeLimited=True,
                         maskFile='/tigress/lthiele/boss_dr12/mask_DR12v5_CMASS_North_nside256.fits',
                         selFunFile=None,
                         zBoundary=(zmin, zmax),
                         zRange=(zmin+0.03, zmax-0.03),
                         minVoidRadius=-1,
                         fakeDensity=fakeDensity,
                         useComoving=True,
                         omegaM=<<<Omega_m>>>)
                 ]
