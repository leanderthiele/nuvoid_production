#!/usr/bin/env python

import os
import os.path
from vide.backend.classes import *

continueRun = False

startCatalogStage = 1
endCatalogStage = 3

AUGMENT = int(os.environ['SLURM_PROCID'])

inputDataDir = '/scratch/gpfs/lthiele/nuvoid_production/test1/galaxies'
workDir = '/scratch/gpfs/lthiele/nuvoid_production/test1/voids'

# it's slightly annoying that VIDE doesn't respect multiprocessing...
logDir = os.path.join(workDir, 'logs%d'%AUGMENT)
figDir = os.path.join(workDir, 'figs%d'%AUGMENT)

numZobovThreads = 8
numZobovDivisions = 2
mergingThreshold = 1e-9

dataSampleList = [Sample(dataFile='lightcone_fidhod_time_samples_%d_augment%d_remap0_stitch1.txt'%(idx, AUGMENT),
                         fullName='time_samples_%d_augment%d_remap0_stitch1'%(idx, AUGMENT),
                         nickName='time_samples_%d_augment%d_remap0_stitch1'%(idx, AUGMENT),
                         dataType='observation',
                         volumeLimited=True, # TODO
                         maskFile='/tigress/lthiele/boss_dr12/mask_DR12v5_CMASS_North_nside256.fits',
                         selFunFile=None, # TODO
                         zBoundary=(0.42, 0.70),
                         zRange=(0.45, 0.67),
                         minVoidRadius=-1,
                         fakeDensity=2e-3,
                         useComoving=True,
                         omegaM=0.3)
                  for idx in range(4,19)]
