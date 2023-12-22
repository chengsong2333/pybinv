import numpy as np
import os.path as op
from BayHunter import SynthObs

# idx = 1
# h = [34, 0]
# vs = [3.5, 4.4]

# idx = 2
# h = [5, 29, 0]
# vs = [3.4, 3.8, 4.5]

# idx = 3
# h = [5, 23, 8, 0]
# vs = [2.7, 3.6, 3.8, 4.4]

idx = 4
h = [0.02, 0.04, 0.09, 0.8, 1, 0]
vs = [0.2, 1, 0.5, 2, 2.5, 3.7]

vpvs = 1.73

path = 'observed'
datafile = op.join(path, 'shallow%d_%s.dat' % (idx, '%s'))

# surface waves
sw_x = np.linspace(0.5, 2.5, 41)
swdata = SynthObs.return_swddata(h, vs, vpvs=vpvs, x=sw_x)
SynthObs.save_data(swdata, outfile=datafile)

# rayleigh wave ellipticity
rwe_x, _ = np.loadtxt('CH001').T
rwedata = SynthObs.return_rwedata(h, vs, vpvs=vpvs, x=rwe_x)
SynthObs.save_data(rwedata, outfile=datafile)

# velocity-depth model
modfile = op.join(path, 'shallow%d_mod.dat' % idx)
SynthObs.save_model(h, vs, vpvs=vpvs, outfile=modfile)
