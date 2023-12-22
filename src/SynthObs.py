# #############################
#
# Cheng Song   (songcheng@snu.ac.kr)
# Modified from Jennifer Dreiling   (dreiling@gfz-potsdam.de)
#
#
# #############################

import numpy as np
import logging
from pybinv import Targets

logger = logging.getLogger()

rstate = np.random.RandomState(333)

def brocher1(vp):
    #     Brocher BSSA 2005, EQ 1
#     1.5 km/s< Vp < 8.5 km/s
#     returns rho in g/cm^3
    vpsq = vp*vp
    vpsq2 = vpsq*vpsq
    rho = 1.6612*vp-0.4721*vpsq+0.0671*vpsq*vp-0.0043*vpsq2+0.000106*vpsq2*vp
    return rho

def brocher2(vs, vp):
#     DM Boore; 
#     http://www.daveboore.com/daves_notes/daves_notes_on_relating_density_to_velocity_v3.0.pdf
#     0 < Vs < 0.3 km/s
#     returns rho in g/cm^3
    if vs<0.3:
        rho = 1. + (1.53*vs**0.85)/(0.35 + 1.889*vs**1.7)
    else:
        rho = 1.74*vp**0.25
    return rho

class SynthObs():
    """SynthObs is a class for computing synthetic 'observed' data.
    Used for testing purposes only. You can compute swd and rf data, and also
    synthetic noise using different correlation laws.
    You find also a method to compute the expected likelihood, for a given
    observed and modeled data set. This value can be feeded into BayWatch."""
    @staticmethod
    def return_swddata(h, vs, vpvs=1.73, pars=dict(), x=None):
        """Return dictionary of forward modeled data based on Surf96."""
        if x is None:
            x = np.linspace(1, 40, 20)

        h = np.array(h)
        vs = np.array(vs)

        mode = pars.get('mode', 1)  # fundamental mode

        target1 = Targets.RayleighDispersionPhase(x=x, y=None)
        target1.moddata.plugin.set_modelparams(mode=mode)
        target2 = Targets.RayleighDispersionGroup(x=x, y=None)
        target2.moddata.plugin.set_modelparams(mode=mode)
        target3 = Targets.LoveDispersionPhase(x=x, y=None)
        target3.moddata.plugin.set_modelparams(mode=mode)
        target4 = Targets.LoveDispersionGroup(x=x, y=None)
        target4.moddata.plugin.set_modelparams(mode=mode)
        vp = vs * vpvs
       # rho = vp * 0.32 + 0.77

        rho = np.ones(len(vs))
        for ii in range(len(rho)):
           if vp[ii] > 1.5:
               rho[ii] = brocher1(vp[ii])
           else:
               rho[ii] = brocher2(vs[ii], vp[ii])


        targets = [target1, target2, target3, target4]

        data = {}
        for i, target in enumerate(targets):
            xmod, ymod = target.moddata.plugin.run_model(
                h=h, vp=vp, vs=vs, rho=rho)
            data[target.ref] = np.array([xmod, ymod])
        logger.info('Compute SWD for %d periods, with model vp/vs %.2f.'
                    % (x.size, vpvs))
        return data

    @staticmethod
    def return_rfdata(h, vs, vpvs=1.73, pars=dict(), x=None):
        """Return dictionary of forward modeled data based on RFMini.
        - x must be linspace to provide an equal sampling rate
        - pars is a dictionary of additional parameters used for RF
        computation (uses defaults if empty), such as:
            - gauss: Gaussian factor (low pass filter),
            - water: water level,
            - p: slowness in s/deg
            - nsv: near surface velocity (km/s) for RF rotation angle.
        """
        if x is None:
            x = np.linspace(-5, 35, 256)

        h = np.array(h)
        vs = np.array(vs)

        gauss = pars['gauss']
        water = pars['water']
        p = pars['p']
        nsv = pars.get('nsv', None)

        target5 = Targets.PReceiverFunction(x=x, y=None)
        target5.moddata.plugin.set_modelparams(
            gauss=gauss, water=water, p=p, nsv=nsv)
        target6 = Targets.SReceiverFunction(x=x, y=None)
        target6.moddata.plugin.set_modelparams(
            gauss=gauss, water=water, p=p, nsv=nsv)

        vp = vs * vpvs
       # rho = vp * 0.32 + 0.77

        rho = np.ones(len(vs))
        for ii in range(len(rho)):
           if vp[ii] > 1.5:
               rho[ii] = brocher1(vp[ii])
           else:
               rho[ii] = brocher2(vs[ii], vp[ii])


        targets = [target5, target6]

        data = {}
        for i, target in enumerate(targets):
            xmod, ymod = target.moddata.plugin.run_model(
                h=h, vp=vp, vs=vs, rho=rho)
            data[target.ref] = np.array([xmod, ymod])
        logger.info('Compute RF with gauss: %.2f, waterlevel: ' % gauss +
                    '%.4f, slowness: %.2f' % (water, p))
        return data

    @staticmethod
    def return_rwedata(h, vs, vpvs=1.73, pars=dict(), x=None):
        """Return dictionary of forward modeled data based on disba."""
        if x is None:
            x = np.linspace(1, 40, 20)

        h = np.array(h)
        vs = np.array(vs)

        mode = pars.get('mode', 1)  # fundamental mode

        target1 = Targets.RayleighWaveEllipticity(x=x, y=None)
        target1.moddata.plugin.set_modelparams(mode=mode)
        vp = vs * vpvs
       # rho = vp * 0.32 + 0.77

        rho = np.ones(len(vs))
        for ii in range(len(rho)):
           if vp[ii] > 1.5:
               rho[ii] = brocher1(vp[ii])
           else:
               rho[ii] = brocher2(vs[ii], vp[ii])


        targets = [target1]

        data = {}
        for i, target in enumerate(targets):
            xmod, ymod = target.moddata.plugin.run_model(
                h=h, vp=vp, vs=vs, rho=rho)
            data[target.ref] = np.array([xmod, ymod])
        logger.info('Compute ellipticity for %d periods, with model vp/vs %.2f.'
                    % (x.size, vpvs))
        return data

    @staticmethod
    def save_data(data, outfile=None):
        """Save data dictionary as ASCII files."""
        if outfile is None:
            outfile = 'syn_%s.dat'

        if '%s' not in outfile:
            name, ext = os.path.splitext(outfile)
            outfile = name + '_%s.'+ext

        for ref in data.keys():
            x, y = data[ref]
            with open(outfile % ref, 'w') as f:
                for i in range(len(x)):
                    f.write('%.4f\t%.4f\n' % (x[i], y[i]))
            logger.info('Data file saved: %s' % outfile % ref)

    @staticmethod
    def save_model(h, vs, vpvs=1.73, outfile=None):
        """Save input model as ASCII file."""
        h = np.array(h)
        vs = np.array(vs)

        vp = vs * vpvs
       # rho = vp * 0.32 + 0.77

        rho = np.ones(len(vs))
        for ii in range(len(rho)):
           if vp[ii] > 1.5:
               rho[ii] = brocher1(vp[ii])
           else:
               rho[ii] = brocher2(vs[ii], vp[ii])


        if outfile is None:
            outfile = 'syn_mod.dat'

        x = np.arange(10)
        target = Targets.PReceiverFunction(x=x, y=None)
        target.moddata.plugin.write_startmodel(h, vp, vs, rho, outfile)
        logger.info('Model file saved: %s' % outfile)

    @staticmethod
    def compute_expnoise(data_obs, corr=0.85, sigma=0.0125):
        """Exponentially correlated noise."""
        idx = np.fromfunction(lambda i, j: (abs((i+j) - 2*i)),
                              (data_obs.size, data_obs.size))
        rmatrix = corr**idx
        Ce = sigma**2 * rmatrix
        data_noise = rstate.multivariate_normal(np.zeros(data_obs.size), Ce)
        return data_noise

    @staticmethod
    def compute_gaussnoise(data_obs, corr=0.85, sigma=0.0125):
        """Gaussian correlated noise - use for RF if Gauss filter applied."""
        idx = np.fromfunction(lambda i, j: (abs((i+j) - 2*i)),
                              (data_obs.size, data_obs.size))
        rmatrix = corr**(idx**2)

        Ce = sigma**2 * rmatrix
        data_noise = rstate.multivariate_normal(np.zeros(data_obs.size), Ce)

        return data_noise

    @staticmethod
    def _nocorr(sigma, size):
        c_inv = np.diag(np.ones(size)) / (sigma**2)
        logc_det = (2*size) * np.log(sigma)
        return c_inv, logc_det

    @staticmethod
    def _gausscorr(sigma, size, corr, rcond=None):
        idx = np.fromfunction(lambda i, j: (abs((i+j) - 2*i)),
                              (size, size))
        rmatrix = corr**(idx**2)

        if rcond is not None:
            corr_inv = np.linalg.pinv(rmatrix, rcond=rcond)
        else:
            corr_inv = np.linalg.inv(rmatrix)

        _, logcorr_det = np.linalg.slogdet(rmatrix)

        c_inv = corr_inv / (sigma**2)
        logc_det = (2*size) * np.log(sigma) + logcorr_det
        return c_inv, logc_det

    @staticmethod
    def _expcorr(sigma, size, corr):
        def get_corr_inv(corr, size):
            d = np.ones(size) + corr**2
            d[0] = d[-1] = 1
            e = np.ones(size-1) * -corr
            corr_inv = np.diag(d) + np.diag(e, k=1) + np.diag(e, k=-1)
            return corr_inv

        c_inv = get_corr_inv(corr, size) / (sigma**2 * (1-corr**2))
        logc_det = (2*size) * np.log(sigma) + (size-1) * np.log(1-corr**2)
        return c_inv, logc_det

    @staticmethod
    def compute_explike(yobss=[], ymods=[], noise=[], gauss=[], rcond=None):
        """Return expected likelihood for observed and modeled data.
        Use for BayWatch only."""
        logL = 0
        for n in range(len(yobss)):
            ydiff = ymods[n] - yobss[n]
            size = ydiff.size

            corr, sigma = noise[2*n:2*n+2]

            if corr == 0:
                c_inv, logc_det = SynthObs._nocorr(sigma, size)
            else:
                if gauss[n]:
                    c_inv, logc_det = SynthObs._gausscorr(
                        sigma, size, corr, rcond=rcond)
                else:
                    c_inv, logc_det = SynthObs._expcorr(sigma, size, corr)

            madist = (ydiff.T).dot(c_inv).dot(ydiff)  # Mahalanobis distance
            logL_part = -0.5 * (size * np.log(2*np.pi) + logc_det)
            logL_target = (logL_part - madist / 2.)
            logL += logL_target

            # print(madist, c_inv, logc_det, logL_target)
            # print(logc_det)
        # print(logL)

        return logL
