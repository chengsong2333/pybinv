# #############################
#
# 
# Cheng Song   (songcheng@snu.ac.kr)
# Modified from Jennifer Dreiling   (dreiling@gfz-potsdam.de)
#
# #############################

import numpy as np
from disba import Ellipticity

class RWE(object):
    """Forward modeling of rayleigh wave ellipticity based on disba (Keurfon Luu).

    https://github.com/keurfonluu/disba
    """

    def __init__(self, obsx, ref):
        self.obsx = obsx
        self.kmax = obsx.size
        self.ref = ref

        self.modelparams = {
            'mode': 1,  # mode, 1 fundamental, 2 first higher
            'flsph': 0  # flat earth model
            }

        self.wavetype, self.veltype = self.get_rwetags(ref)

#         if self.kmax > 60:
#             message = "Your observed data vector exceeds the maximum of 60 \
# periods that is allowed in SurfDisp. For forward modeling SurfDisp will \
# reduce the samples to 60 by linear interpolation within the given period \
# span.\nFrom this data, the dispersion velocities to your observed periods \
# will be determined. The precision of the data will depend on the distribution \
# of your samples and the complexity of the input velocity-depth model."
#             self.obsx_int = np.linspace(obsx.min(), obsx.max(), 60)
#             print(message)

    def set_modelparams(self, **mparams):
        self.modelparams.update(mparams)

    def get_rwetags(self, ref):
        if ref == 'rwe':
            return (1, 1)
        else:
            tagerror = "Reference is not available in SurfDisp. If you defined \
a user Target, assign the correct reference (target.ref) or update the \
forward modeling plugin with target.update_plugin(MyForwardClass()).\n \
* Your ref was: %s\nAvailable refs are: rdispgr, ldispgr, rdispph, ldispph\n \
(r=rayleigh, l=love, gr=group, ph=phase)" % ref
            raise ReferenceError(tagerror)

    def run_model(self, h, vp, vs, rho, **params):
        """ The forward model will be run with the parameters below.

        thkm, vpm, vsm, rhom: model for dispersion calculation
        nlayer - I4: number of layers in the model
        iflsph - I4: 0 flat earth model, 1 spherical earth model
        iwave - I4: 1 Love wave, 2 Rayleigh wave
        mode - I4: ith mode of surface wave, 1 fundamental, 2 first higher, ...
        igr - I4: 0 phase velocity, > 0 group velocity
        kmax - I4: number of periods (t) for dispersion calculation
        t - period vector (t(NP))
        cg - output phase or group velocities (vector,cg(NP))

        """
        # nlayer = len(h)
        vel = np.concatenate((h[:,None], vp[:,None], vs[:,None], rho[:,None]), axis=1)

        # iflsph = self.modelparams['flsph']
        # mode = self.modelparams['mode']
        # iwave = self.wavetype
        # igr = self.veltype

        # if self.kmax > 60:
        #     kmax = 60
        #     pers = self.obsx_int

        # else:
        #     pers = np.zeros(60)
        #     kmax = self.kmax
        pers = self.obsx

        # dispvel = np.zeros(60)  # result
        try:
            ell = Ellipticity(*vel.T)
            rel_0 = ell(pers, mode=0)

        # if error == 0:
            # if self.kmax > 60:
            #     disp_int = np.interp(self.obsx, pers, dispvel)
                # return self.obsx, disp_int
            return pers, np.abs(rel_0.ellipticity)
        except:
            return np.nan, np.nan
