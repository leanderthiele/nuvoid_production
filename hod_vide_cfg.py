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

numZobovThreads = 8
numZobovDivisions = 2
mergingThreshold = 1e-9

zmin = <<<zmin>>>
zmax = <<<zmax>>>

dataSampleList = [Sample(dataFile='lightcone_<<<augment>>>.txt',
                         fullName='<<<augment>>>',
                         nickName='<<<augment>>>',
                         dataType='observation',
                         volumeLimited=True, # TODO
                         maskFile='/tigress/lthiele/boss_dr12/mask_DR12v5_CMASS_North_nside256.fits',
                         selFunFile=None, # TODO
                         zBoundary=(zmin, zmax),
                         zRange=(zmin+0.02, zmax-0.02),
                         minVoidRadius=-1,
                         fakeDensity=2e-3,
                         useComoving=True,
                         omegaM=<<<Omega_m>>>)
                 ]
